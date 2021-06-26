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


class DIV2KCommonConfig(BaseConfig):
    """Default Dataset config for DIV2K."""

    root_HR = None
    root_LR = None
    num_workers = 4
    upscale = 2
    subfile = None  # Set it to None by default
    crop = None  # Set it to None by default
    hflip = False
    vflip = False
    rot90 = False
    save_in_memory = False
    value_div = 1.0

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Common_DIV2K = {"upscale": {"type": int},
                              "root_HR": {"type": str},
                              "root_LR": {"type": str},
                              "num_workers": {"type": int},
                              "subfile": {"type": (str, None)},
                              "crop": {"type": (int, None)},
                              "hflip": {"type": bool},
                              "vflip": {"type": bool},
                              "rot90": {"type": bool},
                              "save_in_memory": {"type": bool},
                              "value_div": {"type": float}
                              }
        return rules_Common_DIV2K


class DIV2KTrainConfig(DIV2KCommonConfig):
    """Default Dataset config for DIV2K."""

    load_size = 1024
    crop_size = 120

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Train_DIV2K = {"upscale": {"type": int},
                             "load_size": {"type": int},
                             "crop_size": {"type": int},
                             "root_HR": {"type": str},
                             "root_LR": {"type": str},
                             "num_workers": {"type": int},
                             "subfile": {"type": (str, None)},
                             "crop": {"type": (int, None)},
                             "hflip": {"type": bool},
                             "vflip": {"type": bool},
                             "rot90": {"type": bool}
                             }
        return rules_Train_DIV2K


class DIV2KTestConfig(DIV2KCommonConfig):
    """Default Dataset config for DIV2K."""

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Test_DIV2K = {"upscale": {"type": int},
                            "load_size": {"type": int},
                            "crop_size": {"type": int},
                            "root_HR": {"type": str},
                            "root_LR": {"type": str},
                            "num_workers": {"type": int},
                            "subfile": {"type": (str, None)},
                            "crop": {"type": (int, None)},
                            "hflip": {"type": bool},
                            "vflip": {"type": bool},
                            "rot90": {"type": bool}
                            }
        return rules_Test_DIV2K


class DIV2KConfig(ConfigSerializable):
    """Default Dataset config for DIV2K."""

    common = DIV2KCommonConfig
    train = DIV2KTrainConfig
    test = DIV2KTestConfig

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_DIV2K = {"type": {"type": str},
                       "common": {"type": dict},
                       "train": {"type": dict},
                       "test": {"type": dict}
                       }
        return rules_DIV2K

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {'common': cls.common,
                'train': cls.train,
                'test': cls.test
                }
