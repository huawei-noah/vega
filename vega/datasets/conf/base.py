# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""
from vega.common import ConfigSerializable


class BaseConfig(ConfigSerializable):
    """Base config of dataset."""

    data_path = None
    batch_size = 1
    num_workers = 0
    imgs_per_gpu = 1,
    shuffle = False
    distributed = False
    download = False
    pin_memory = True
    drop_last = True
    transforms = []
    buffer_size = 128

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Base = {"data_path": {"type": (str, None)},
                      "batch_size": {"type": int},
                      "num_workers": {"type": int},
                      "shuffle": {"type": bool},
                      "distributed": {"type": bool},
                      "download": {"type": bool},
                      "pin_memory": {"type": bool},
                      "drop_last": {"type": bool},
                      "transforms": {"type": list},
                      }
        return rules_Base


class ExtConfig(BaseConfig):
    """Extension config."""

    def __getattr__(self, item):
        """Override getattr function."""
        if hasattr(self, item):
            return super().__getattribute__(item)
        else:
            return None


class ExtDatasetConfig(ConfigSerializable):
    """Extension dataset config."""

    common = ExtConfig
    train = ExtConfig
    val = ExtConfig
    test = ExtConfig
