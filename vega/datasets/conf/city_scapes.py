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


class CityscapesCommonConfig(BaseConfig):
    """Default Dataset config for Cityscapes."""

    batch_size = 1
    root_path = None
    num_parallel_batches = 64
    fixed_size = True
    train_portion = 1.0

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_CityscapesConfig = {"batch_size": {"type": int},
                                  "root_path": {"type": str},
                                  "num_parallel_batches": {"type": int},
                                  "fixed_size": {"type": bool}
                                  }
        return rules_CityscapesConfig


class CityscapesTrainConfig(CityscapesCommonConfig):
    """Default Dataset config for Cityscapes."""

    batch_size = 1
    list_path = 'train.txt'

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_CityscapesTrainConfig = {"batch_size": {"type": int},
                                       "list_path": {"type": str}
                                       }
        return rules_CityscapesTrainConfig


class CityscapesValConfig(CityscapesCommonConfig):
    """Default Dataset config for Cityscapes."""

    batch_size = 1
    list_path = 'val.txt'

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_CityscapesValConfig = {"batch_size": {"type": int},
                                     "list_path": {"type": str}
                                     }
        return rules_CityscapesValConfig


class CityscapesTestConfig(CityscapesCommonConfig):
    """Default Dataset config for Cityscapes."""

    batch_size = 1
    list_path = 'val.txt'

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_CityscapesTestConfig = {"batch_size": {"type": int},
                                      "list_path": {"type": str}
                                      }
        return rules_CityscapesTestConfig


class CityscapesConfig(ConfigSerializable):
    """Default Dataset config for Cityscapes."""

    common = CityscapesCommonConfig
    train = CityscapesTrainConfig
    val = CityscapesValConfig
    test = CityscapesTestConfig

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Cityscapes = {"common": {"type": dict},
                            "train": {"type": dict},
                            "val": {"type": dict},
                            "test": {"type": dict}
                            }
        return rules_Cityscapes

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {'common': cls.common,
                'train': cls.train,
                'val': cls.val,
                'test': cls.test
                }
