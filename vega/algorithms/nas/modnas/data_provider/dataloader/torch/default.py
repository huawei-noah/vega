# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default DataLoader."""

from typing import Any, Dict, Optional, Tuple, Union, Callable
import random
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from modnas.registry.data_loader import register
from modnas.utils.logging import get_logger


logger = get_logger('data_loader')


@register
def DefaultDataLoader(
        trn_data: Dataset,
        val_data: Optional[Dataset],
        trn_batch_size: int = 64,
        val_batch_size: int = 64,
        workers: int = 2,
        collate_fn: Optional[Callable] = None,
        parallel_multiplier: int = 1,
        train_size: int = 0,
        train_ratio: float = 1.,
        train_seed: int = 1,
        valid_size: int = 0,
        valid_ratio: Union[float, int] = 0.,
        valid_seed: int = 1,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
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
    extra_kwargs: Dict[str, Any] = {
        'num_workers': workers,
        'pin_memory': True,
    }
    if collate_fn is not None:
        extra_kwargs['collate_fn'] = collate_fn
    if len(trn_idx) > 0:
        trn_sampler = SubsetRandomSampler(trn_idx)
        trn_loader = DataLoader(trn_data, batch_size=trn_batch_size, sampler=trn_sampler, **extra_kwargs)
    if len(val_idx) > 0:
        val_sampler = SubsetRandomSampler(val_idx)
        val_loader = DataLoader(val_data, batch_size=val_batch_size, sampler=val_sampler, **extra_kwargs)
    return trn_loader, val_loader
