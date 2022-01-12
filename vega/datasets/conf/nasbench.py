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
