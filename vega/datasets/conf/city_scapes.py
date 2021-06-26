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
