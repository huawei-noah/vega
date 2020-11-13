# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""

from .base import BaseConfig
from zeus.common import ConfigSerializable


class CityscapesCommonConfig(BaseConfig):
    """Default Dataset config for Cityscapes."""

    batch_size = 1
    root_path = None
    num_parallel_batches = 64
    fixed_size = True


class CityscapesTrainConfig(CityscapesCommonConfig):
    """Default Dataset config for Cityscapes."""

    batch_size = 1
    list_path = 'train.txt'


class CityscapesValConfig(CityscapesCommonConfig):
    """Default Dataset config for Cityscapes."""

    batch_size = 1
    list_path = 'val.txt'


class CityscapesTestConfig(CityscapesCommonConfig):
    """Default Dataset config for Cityscapes."""

    batch_size = 1
    list_path = 'val.txt'


class CityscapesConfig(ConfigSerializable):
    """Default Dataset config for Cityscapes."""

    common = CityscapesCommonConfig
    train = CityscapesTrainConfig
    val = CityscapesValConfig
    test = CityscapesTestConfig
