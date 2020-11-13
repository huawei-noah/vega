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
from zeus.common import ConfigSerializable


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


class SRTestConfig(SRCommonConfig):
    """Default Dataset config for SR."""

    pass


class SRConfig(ConfigSerializable):
    """Default Dataset config for SR."""

    common = SRCommonConfig
    test = SRTestConfig
