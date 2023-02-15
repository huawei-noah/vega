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

import importlib
from vega.common import ClassFactory, ClassType, TaskOps
from vega.common.config import Config
from vega.common import update_dict
from vega.datasets import Adapter
from .utils.transforms import Transforms


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
        self.frame_type = False
        dataset_backend = self.args.get("dataset_backend")
        if dataset_backend in ["m", "mindspore"]:
            self.frame_type = True
        if not self.frame_type:
            transforms_list = self._init_transforms()
            self._transforms = Transforms(transforms_list)
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
