# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a base class of the dataset."""

import importlib
from vega.common.task_ops import TaskOps
from .utils.transforms import Transforms
from vega.common import ClassFactory, ClassType
from vega.common.config import Config
from vega.common import update_dict
from vega.datasets import Adapter


class Dataset(TaskOps):
    """This is the base class of the dataset, which is a subclass of `TaskOps`.

    The Dataset provide several basic attribute like dataloader, transform and sampler.
    """

    def __new__(cls, *args, **kwargs):
        """Create a subclass instance of dataset."""
        if Dataset in cls.__bases__:
            return super().__new__(cls)
        if kwargs.get('type'):
            t_cls = ClassFactory.get_cls(ClassType.DATASET, kwargs.pop('type'))
        else:
            t_cls = ClassFactory.get_cls(ClassType.DATASET)
        return super().__new__(t_cls)

    def __init__(self, hps=None, mode='train', **kwargs):
        """Construct method."""
        super(Dataset, self).__init__()
        self.args = dict()
        self.mode = mode
        if mode == "val" and not hasattr(self.config, "val") and not hasattr(self.config.common,
                                                                             "train_portion"):
            self.mode = "test"
        # modify config from kwargs, `Cifar10(mode='test', data_path='/cache/datasets')`
        if kwargs:
            self.args = Config(kwargs)
        if hasattr(self, 'config'):
            if hps is not None:
                self.config.from_dict(hps)
            config = getattr(self.config, self.mode)
            config.from_dict(self.args)
            self.args = config().to_dict()
        self._init_hps(hps)
        self.train = self.mode in ["train", "val"]
        transforms_list = self._init_transforms()
        self._transforms = Transforms(transforms_list)
        # if "transforms" in kwargs.keys():
        #     self._transforms.__transform__ = kwargs["transforms"]
        self.dataset_init()
        self.world_size = 1
        self.rank = 0
        self.collate_fn = None

    def dataset_init(self):
        """Init Dataset before sampler."""
        pass

    def _init_hps(self, hps):
        """Convert trainer values in hps to cfg."""
        if hps is not None:
            self.args = Config(update_dict(hps, self.args))

    @property
    def transforms(self):
        """Transform function which can replace transforms."""
        return self._transforms

    @transforms.setter
    def transforms(self, value):
        """Set function of transforms."""
        self._transforms = value

    def _init_transforms(self):
        """Initialize transforms method.

        :return: a list of object
        :rtype: list
        """
        if "transforms" in self.args.keys():
            transforms = list()
            if not isinstance(self.args.transforms, list):
                self.args.transforms = [self.args.transforms]
            for i in range(len(self.args.transforms)):
                transform_name = self.args.transforms[i].pop("type")
                kwargs = self.args.transforms[i]
                try:
                    transforms.append(getattr(importlib.import_module('torchvision.transforms'),
                                              transform_name)(**kwargs))
                except Exception:
                    transforms.append(ClassFactory.get_cls(ClassType.TRANSFORM, transform_name)(**kwargs))

            return transforms
        else:
            return list()

    def __len__(self):
        """Get the length of the dataset."""
        raise NotImplementedError

    def __getitem__(self, index):
        """Get an item of the dataset according to the index."""
        raise NotImplementedError

    def set_distributed(self, world_size, rank):
        """Set distribued parameters."""
        self.world_size = world_size
        self.rank = rank

    @property
    def loader(self):
        """Return loader."""
        return Adapter(self).loader
