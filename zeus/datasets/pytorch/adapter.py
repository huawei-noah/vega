# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a base class of the dataset."""
from torch.utils import data as torch_data
from .samplers import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


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
            self.args.shuffle = False
            sampler = DistributedSampler(self.dataset,
                                         num_replicas=self.dataset.world_size,
                                         rank=self.dataset.rank,
                                         shuffle=self.args.shuffle)
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
        data_loader = torch_data.DataLoader(dataset=self.dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=self.args.shuffle,
                                            num_workers=self.args.num_workers,
                                            pin_memory=self.args.pin_memory,
                                            sampler=self.sampler,
                                            drop_last=self.args.drop_last,
                                            collate_fn=self.collate_fn)
        return data_loader
