# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""
from zeus.common import ConfigSerializable


class BaseConfig(ConfigSerializable):
    """Base config of dataset."""

    data_path = None
    batch_size = 1
    num_workers = 0
    imgs_per_gpu = 1,
    shuffle = False
    distributed = False
    download = False
    pin_memory = True
    drop_last = True
    transforms = []
