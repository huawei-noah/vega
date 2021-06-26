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


class Cifar100CommonConfig(BaseConfig):
    """Default Cifar100 Config."""

    n_class = 100
    batch_size = 256
    num_workers = 4
    train_portion = 1.0

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Cifar100CommonConfig = {"n_class": {"type": int},
                                      "batch_size": {"type": int},
                                      "num_workers": {"type": int}
                                      }
        return rules_Cifar100CommonConfig


class Cifar100TrainConfig(Cifar100CommonConfig):
    """Default Cifar100 config."""

    transforms = [
        dict(type='RandomCrop', size=32, padding=4),
        dict(type='RandomHorizontalFlip'),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.50707519, 0.48654887, 0.44091785], std=[0.26733428, 0.25643846, 0.27615049])]

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Cifar100TrainConfig = {
            "transforms": {"type": list}
        }
        return rules_Cifar100TrainConfig


class Cifar100ValConfig(Cifar100CommonConfig):
    """Default Cifar100 config."""

    transforms = [
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.50707519, 0.48654887, 0.44091785], std=[0.26733428, 0.25643846, 0.27615049])]

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Cifar100ValConfig = {
            "transforms": {"type": list}
        }
        return rules_Cifar100ValConfig


class Cifar100TestConfig(Cifar100CommonConfig):
    """Default Cifar100 config."""

    transforms = [
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.50707519, 0.48654887, 0.44091785], std=[0.26733428, 0.25643846, 0.27615049])]

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Cifar100TestConfig = {
            "transforms": {"type": list}
        }
        return rules_Cifar100TestConfig


class Cifar100Config(ConfigSerializable):
    """Default Dataset config for Cifar100."""

    common = Cifar100CommonConfig
    train = Cifar100TrainConfig
    val = Cifar100ValConfig
    test = Cifar100TestConfig

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Cifar100 = {"common": {"type": dict},
                          "train": {"type": dict},
                          "val": {"type": dict},
                          "test": {"type": dict}
                          }
        return rules_Cifar100

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {'common': cls.common,
                'train': cls.train,
                'val': cls.val,
                'test': cls.test
                }
