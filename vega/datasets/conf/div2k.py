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
