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


class MnistCommonConfig(BaseConfig):
    """Default Dataset config for Mnist."""

    n_class = 10


class MnistTrainConfig(MnistCommonConfig):
    """Default Dataset config for Mnist."""

    transforms = [dict(type='RandomAffine', degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[0.13066051707548254], std=[0.30810780244715075])]


class MnistValConfig(MnistCommonConfig):
    """Default Dataset config for Mnist."""

    transforms = [dict(type='ToTensor'),
                  dict(type='Normalize', mean=[0.13066051707548254], std=[0.30810780244715075])]


class MnistTestConfig(MnistCommonConfig):
    """Default Dataset config for Mnist."""

    transforms = [dict(type='ToTensor'),
                  dict(type='Normalize', mean=[0.13066051707548254], std=[0.30810780244715075])]


class MnistConfig(ConfigSerializable):
    """Default Dataset config for Mnist."""

    common = MnistCommonConfig
    train = MnistTrainConfig
    val = MnistValConfig
    test = MnistTestConfig
