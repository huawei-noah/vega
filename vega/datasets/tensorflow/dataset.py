# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a base class of the dataset."""
from copy import deepcopy
from vega.core.common.task_ops import TaskOps
from vega.core.common.config import Config, obj2config
from vega.core.common.utils import update_dict


class Dataset(TaskOps):
    """This is the base class of the dataset, which is a subclass of `TaskOps`.

    The Dataset provide several basic attribute like dataloader, transform and sampler.
    """

    def __init__(self, hps=None, mode='train', **kwargs):
        """Construct method."""
        super(Dataset, self).__init__()
        if not hasattr(self, 'config'):
            raise ValueError("Dataset class should has attr config.")
        self.mode = mode
        if self.mode == "test" and not hasattr(self.config, "test"):
            self.mode = "val"
        self.args = deepcopy(obj2config(getattr(self.config, self.mode)))
        self._init_hps(hps)
        self.train = self.mode in ["train", "val"]
        self.num_images = self.args.get('num_images', 0)
        self.batch_size = self.args.batch_size
        self.world_size = 1
        self.rank = 0

    def _init_hps(self, hps):
        """Convert trainer values in hps to cfg."""
        if hps is not None:
            self.args = Config(update_dict(hps, self.args))

    def __len__(self):
        """Return dataset length of train or valid."""
        if self.mode == 'train':
            len = self.num_images // self.batch_size
            if self.world_size > 1:
                len = len // self.world_size
        else:
            len = self.num_images // self.batch_size
        return len

    def set_distributed(self, world_size, rank):
        """Set distribued parameters."""
        self.world_size = world_size
        self.rank = rank
