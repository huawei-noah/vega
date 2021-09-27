# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Create train or eval dataset."""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size


def create_dataset2(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False,
                    enable_cache=False, cache_session_id=None):
    """
    Create a train or eval imagenet2012 dataset for resnet50.

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False
        enable_cache(bool): whether tensor caching service is used for eval. Default: False
        cache_session_id(int): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num = int(os.environ.get('RANK_SIZE'))
        rank_id = int(os.environ.get('RANK_ID'))
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1

    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=12, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=12, shuffle=True,
                                         num_shards=device_num, shard_id=rank_id)

    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=12)
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12,
                                cache=eval_cache)
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
