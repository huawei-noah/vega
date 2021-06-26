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


class NasbenchCommonConfig(BaseConfig):
    """Common nasbench Config."""

    batch_size = 64
    epochs = 108
    computed_key = 'final_test_accuracy'
    portion = 0.95

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_NasbenchCommonConfig = {"epochs": {"type": int},
                                      "computed_key": {"type": str},
                                      "batch_size": {"type": int},
                                      }
        return rules_NasbenchCommonConfig


class NasbenchTrainConfig(NasbenchCommonConfig):
    """Default Cifar10 config."""

    pass


class NasbenchValConfig(NasbenchCommonConfig):
    """Default Cifar10 config."""

    pass


class NasbenchTestConfig(NasbenchCommonConfig):
    """Default nasbench test config."""

    pass


class NasbenchConfig(ConfigSerializable):
    """Default Dataset config for Cifar10."""

    common = NasbenchCommonConfig
    train = NasbenchTrainConfig
    val = NasbenchValConfig
    test = NasbenchTestConfig

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Cifar10 = {"common": {"type": dict},
                         "train": {"type": dict},
                         "val": {"type": dict},
                         "test": {"type": dict}
                         }
        return rules_Cifar10

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {'common': cls.common,
                'train': cls.train,
                'val': cls.val,
                'test': cls.test
                }
