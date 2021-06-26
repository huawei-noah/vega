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
from vega.common import ConfigSerializable


class MnistCommonConfig(BaseConfig):
    """Default Dataset config for Mnist."""

    n_class = 10
    train_portion = 1.0

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_MnistCommon = {"n_class": {"type": int}}
        return rules_MnistCommon


class MnistTrainConfig(MnistCommonConfig):
    """Default Dataset config for Mnist."""

    transforms = [dict(type='RandomAffine', degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[0.13066051707548254], std=[0.30810780244715075])]

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_MnistTrain = {"transforms": {"type": list}}
        return rules_MnistTrain


class MnistValConfig(MnistCommonConfig):
    """Default Dataset config for Mnist."""

    transforms = [dict(type='ToTensor'),
                  dict(type='Normalize', mean=[0.13066051707548254], std=[0.30810780244715075])]

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_MnistVal = {"transforms": {"type": list}}
        return rules_MnistVal


class MnistTestConfig(MnistCommonConfig):
    """Default Dataset config for Mnist."""

    transforms = [dict(type='ToTensor'),
                  dict(type='Normalize', mean=[0.13066051707548254], std=[0.30810780244715075])]

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_MnistTest = {"transforms": {"type": list}}
        return rules_MnistTest


class MnistConfig(ConfigSerializable):
    """Default Dataset config for Mnist."""

    common = MnistCommonConfig
    train = MnistTrainConfig
    val = MnistValConfig
    test = MnistTestConfig

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Mnist = {"common": {"type": dict},
                       "train": {"type": dict},
                       "val": {"type": dict},
                       "test": {"type": dict}
                       }
        return rules_Mnist

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {'common': cls.common,
                'train': cls.train,
                'val': cls.val,
                'test': cls.test
                }
