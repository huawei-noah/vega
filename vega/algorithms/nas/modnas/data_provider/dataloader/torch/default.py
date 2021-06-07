# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Default DataLoader."""
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from modnas.registry.data_loader import register
from modnas.utils.logging import get_logger


logger = get_logger('data_loader')


@register
def DefaultDataLoader(trn_data,
                      val_data,
                      parallel_multiplier=1,
                      trn_batch_size=64,
                      val_batch_size=64,
                      workers=2,
                      train_size=0,
                      train_ratio=1.,
                      train_seed=1,
                      valid_size=0,
                      valid_ratio=0.,
                      valid_seed=1):
    """Return default DataLoader."""
    # index
    n_train_data = len(trn_data)
    trn_idx = list(range(n_train_data))
    if train_size <= 0:
        train_size = int(n_train_data * min(train_ratio, 1.))
    if 0 < train_size < n_train_data:
        random.seed(train_seed)
        trn_idx = random.sample(trn_idx, train_size)
    if val_data is not None:
        n_valid_data = len(val_data)
        val_idx = list(range(n_valid_data))
        if valid_size <= 0 and valid_ratio > 0:
            valid_size = int(n_valid_data * min(valid_ratio, 1.))
        if 0 < valid_size < n_valid_data:
            random.seed(valid_seed)
            val_idx = random.sample(val_idx, valid_size)
    else:
        val_data = trn_data
        if valid_size <= 0 and valid_ratio > 0:
            valid_size = int(train_size * min(valid_ratio, 1.))
        if valid_size > 0:
            random.seed(valid_seed)
            random.shuffle(trn_idx)
            val_idx, trn_idx = trn_idx[:valid_size], trn_idx[valid_size:]
        else:
            val_idx = list()
    logger.info('data_loader: trn: {} val: {}'.format(len(trn_idx), len(val_idx)))
    # dataloader
    trn_loader = val_loader = None
    trn_batch_size *= parallel_multiplier
    val_batch_size *= parallel_multiplier
    workers *= parallel_multiplier
    extra_kwargs = {
        'num_workers': workers,
        'pin_memory': True,
    }
    if len(trn_idx) > 0:
        trn_sampler = SubsetRandomSampler(trn_idx)
        trn_loader = DataLoader(trn_data, batch_size=trn_batch_size, sampler=trn_sampler, **extra_kwargs)
    if len(val_idx) > 0:
        val_sampler = SubsetRandomSampler(val_idx)
        val_loader = DataLoader(val_data, batch_size=val_batch_size, sampler=val_sampler, **extra_kwargs)
    return trn_loader, val_loader
