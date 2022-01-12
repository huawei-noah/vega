# -*- coding: utf-8 -*-

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

"""This is a base class of the dataset."""
import logging

from torch.utils import data as torch_data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from .samplers import DistributedSampler


class TorchAdapter(object):
    """This is the base class of the dataset, which is a subclass of `TaskOps`.

    The Dataset provide several basic attribute like dataloader, transform and sampler.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.args = dataset.args
        self.sampler = self._init_sampler()
        self.collate_fn = dataset.collate_fn

    @property
    def sampler(self):
        """Sampler function which can replace sampler."""
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        """Set function of sampler."""
        self._sampler = value

    def _init_sampler(self):
        """Initialize sampler method.

        :return: if the distributed is True, return a sampler object, else return None
        :rtype: an object or None
        """
        if self.dataset.world_size > 1:
            sampler = DistributedSampler(self.dataset,
                                         num_replicas=self.dataset.world_size,
                                         rank=self.dataset.rank,
                                         shuffle=self.args.shuffle)
            self.args.shuffle = False
        elif not hasattr(self.args, "train_portion"):
            sampler = None
        elif self.dataset.mode == 'test' or self.args.train_portion == 1:
            sampler = None
        else:
            self.args.shuffle = False
            num_train = len(self.dataset)
            indices = list(range(num_train))
            split = int(np.floor(self.args.train_portion * num_train))
            if self.dataset.mode == 'train':
                sampler = SubsetRandomSampler(indices[:split])
            elif self.dataset.mode == 'val':
                sampler = SubsetRandomSampler(indices[split:num_train])
            else:
                raise ValueError('the mode should be train, val or test')
        return sampler

    @property
    def loader(self):
        """Dataloader arrtribute which is a unified interface to generate the data.

        :return: a batch data
        :rtype: dict, list, optional
        """
        if hasattr(self.dataset, "data_loader"):
            return self.dataset.data_loader
        try:
            data_loader = torch_data.DataLoader(dataset=self.dataset,
                                                batch_size=self.args.batch_size,
                                                shuffle=self.args.shuffle,
                                                num_workers=self.args.num_workers,
                                                pin_memory=self.args.pin_memory,
                                                sampler=self.sampler,
                                                drop_last=self.args.drop_last,
                                                collate_fn=self.collate_fn)
        except BrokenPipeError as ex:
            logging.debug(ex)
            data_loader = None
        return data_loader
