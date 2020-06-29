# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Training with mmdet."""

from __future__ import division
import argparse
import os
import torch
from mmcv import Config
import mmcv
from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from vega.algorithms.nas.sp_nas.spnet import *
from vega.algorithms.nas.sp_nas.utils.config_utils import json_to_dict


def parse_args():
    """Get input."""
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    """Start train."""
    args = parse_args()
    if args.config.endswith(".json"):
        load_method = mmcv.load
        mmcv.load = json_to_dict
        cfg = mmcv.Config.fromfile(args.config)
        mmcv.load = load_method
    else:
        cfg = mmcv.Config.fromfile(args.config)
    cfg = Config.fromfile(args.config)
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # init distributed env， logger and model.
    init_dist('pytorch', **cfg.dist_params)
    logger = get_root_logger(cfg.log_level)
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=True,
        logger=logger)


if __name__ == '__main__':
    main()
