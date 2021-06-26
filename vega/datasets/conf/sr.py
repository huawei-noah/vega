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
