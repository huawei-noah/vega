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


class Cifar10CommonConfig(BaseConfig):
    """Default Optim Config."""

    n_class = 10
    batch_size = 256
    num_workers = 8
    train_portion = 1.0
    num_parallel_batches = 64
    fp16 = False

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Cifar10CommonConfig = {"n_class": {"type": int},
                                     "batch_size": {"type": int},
                                     "num_workers": {"type": int},
                                     "train_portion": {"type": (int, float)},
                                     "num_parallel_batches": {"type": int},
                                     "fp16": {"type": bool}
                                     }
        return rules_Cifar10CommonConfig


class Cifar10TrainConfig(Cifar10CommonConfig):
    """Default Cifar10 config."""

    transforms = [
        dict(type='RandomCrop', size=32, padding=4),
        dict(type='RandomHorizontalFlip'),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])]
    padding = 8
    num_images = 50000

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Cifar10TrainConfig = {"padding": {"type": int},
                                    "num_images": {"type": int},
                                    "transforms": {"type": list}
                                    }
        return rules_Cifar10TrainConfig


class Cifar10ValConfig(Cifar10CommonConfig):
    """Default Cifar10 config."""

    transforms = [
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])]
    num_images = 10000
    num_images_train = 50000

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Cifar10ValConfig = {"num_images": {"type": int},
                                  "num_images_train": {"type": int},
                                  "transforms": {"type": list}
                                  }
        return rules_Cifar10ValConfig


class Cifar10TestConfig(Cifar10CommonConfig):
    """Default Cifar10 config."""

    transforms = [
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])]
    num_images = 10000

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Cifar10TestConfig = {"num_images": {"type": int},
                                   "transforms": {"type": list}
                                   }
        return rules_Cifar10TestConfig


class Cifar10Config(ConfigSerializable):
    """Default Dataset config for Cifar10."""

    common = Cifar10CommonConfig
    train = Cifar10TrainConfig
    val = Cifar10ValConfig
    test = Cifar10TestConfig

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
