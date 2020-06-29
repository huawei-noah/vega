# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Testing with mmdet."""

import argparse
import os
import os.path as osp
import shutil
import tempfile
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import get_dist_info, load_checkpoint
from mmdet.apis import init_dist
from mmdet.core import results2json
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from vega.algorithms.nas.sp_nas.utils import Timer, coco_eval
from vega.algorithms.nas.sp_nas.spnet import *
from vega.algorithms.nas.sp_nas.utils.config_utils import json_to_dict


def single_gpu_test(model, data_loader, dump_file=None):
    """Single gpu test.

    :param model: test model
    :type model: model
    :param data_loader: data loader
    :type data_loader: data_loader
    :param dump_file: dump file dir
    :type dump_file: str
    :return: predict results
    :rtype: list
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    t_data, t_model = 0, 0
    timer = Timer(cuda_mode=True)
    for i, data in enumerate(data_loader):
        t_data += timer.since_last_check()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        t_model += timer.since_last_check()
        results.append(result)

    fps = dict(model=len(dataset) / t_model,
               data=len(dataset) / t_data,
               total=len(dataset) / (t_model + t_data))
    print('fps_model:{model}; fps_data:{data}; fps_total:{total}'.format(**fps))
    if dump_file is not None:
        mmcv.dump(fps, dump_file)
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    """Multiple gpu test.

    :param model: test model
    :type model: model
    :param data_loader: data loader
    :type data_loader: data_loader
    :param tmpdir: tmp file dir
    :type tmpdir: str
    :return: predict results
    :rtype: list
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    """Collect results for multi-gpu test.

    :param result_part: part of results
    :type result_part: list
    :param size: number of data
    :type size: int
    :param tmpdir: tmp file dir
    :type tmpdir: str
    :return: predict results
    :rtype: list
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    """Get input settings.

    :return: args
    :rtype: args
    """
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--work_dir', help='work path')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--eval', type=str, nargs='+', default=['bbox'],
                        choices=['bbox', 'segm'], help='eval types')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', help='Whether distributed test (multi-gpu test)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():  # noqa: C901
    """Start test."""
    args = parse_args()

    if args.work_dir is not None:
        mmcv.mkdir_or_exist(args.work_dir)
        if args.tmpdir is None:
            args.tmpdir = osp.join(args.work_dir, 'tmp_dir')
            mmcv.mkdir_or_exist(args.tmpdir)
        if args.out is None:
            args.out = osp.join(args.work_dir, 'result.pkl')
        if args.checkpoint is None:
            args.checkpoint = osp.join(args.work_dir, 'latest.pth')
        fps_file = osp.join(args.work_dir, 'fps.pkl')
        mAP_file = osp.join(args.work_dir, 'mAP.pkl')
    else:
        mAP_file, fps_file = None, None
        if args.checkpoint is None:
            raise ValueError('Checkpoint file cannot be empty.')

    if args.config.endswith(".json"):
        load_method = mmcv.load
        mmcv.load = json_to_dict
        cfg = mmcv.Config.fromfile(args.config)
        mmcv.load = load_method
    else:
        cfg = mmcv.Config.fromfile(args.config)
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.dist:
        init_dist('pytorch', **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    if args.dist:
        model = MMDistributedDataParallel(model.cuda(),
                                          device_ids=[torch.cuda.current_device()],
                                          broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)
    else:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, fps_file)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            if eval_types:
                print('Starting evaluate {}'.format(' and '.join(eval_types)))
                assert not isinstance(outputs[0], dict)
                result_files = results2json(dataset, outputs, args.out)
                coco_eval(result_files, eval_types, dataset.coco, dump_file=mAP_file)


if __name__ == '__main__':
    main()
