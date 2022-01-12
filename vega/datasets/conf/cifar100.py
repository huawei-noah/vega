# -*- coding=utf-8 -*-

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
"""Default configs."""

from vega.common import ConfigSerializable
from .base import BaseConfig


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
