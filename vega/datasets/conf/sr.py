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


class SRCommonConfig(BaseConfig):
    """Default Dataset config for SR."""

    root_HR = None
    root_LR = None
    num_workers = 4
    upscale = 2
    subfile = None
    crop = None
    hflip = False
    vflip = False
    rot90 = False
    save_in_memory = False

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_SRCommon = {"root_HR": {"type": str},
                          "root_LR": {"type": str},
                          "num_workers": {"type": int},
                          "upscale": {"type": int},
                          "subfile": {"type": (str, None)},
                          "crop": {"type": int},
                          "hflip": {"type": bool},
                          "vflip": {"type": bool},
                          "rot90": {"type": bool},
                          "save_in_memory": {"type": bool}
                          }
        return rules_SRCommon


class SRTestConfig(SRCommonConfig):
    """Default Dataset config for SR."""

    pass


class SRConfig(ConfigSerializable):
    """Default Dataset config for SR."""

    common = SRCommonConfig
    test = SRTestConfig

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_SR = {"common": {"type": dict},
                    "test": {"type": dict}
                    }
        return rules_SR

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {'common': cls.common,
                'test': cls.test
                }
