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
        if hasattr(self.dataset, '_init_sampler'):
            return self.dataset._init_sampler()

        if self.dataset.world_size > 1:
            self.args.shuffle = False
            sampler = DistributedSampler(self.dataset,
                                         num_replicas=self.dataset.world_size,
                                         rank=self.dataset.rank,
                                         shuffle=self.args.shuffle)
        else:
            sampler = None
        return sampler

    @property
    def loader(self):
        """Dataloader arrtribute which is a unified interface to generate the data.

        :return: a batch data
        :rtype: dict, list, optional
        """
        if hasattr(self.dataset, "loader"):
            return self.dataset.loader
        data_loader = torch_data.DataLoader(dataset=self.dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=self.args.shuffle,
                                            num_workers=self.args.num_workers,
                                            pin_memory=self.args.pin_memory,
                                            sampler=self.sampler,
                                            drop_last=self.args.drop_last,
                                            collate_fn=self.collate_fn)
        return data_loader
