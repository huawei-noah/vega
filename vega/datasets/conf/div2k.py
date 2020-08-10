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


class DIV2KTrainConfig(DIV2KCommonConfig):
    """Default Dataset config for DIV2K."""

    load_size = 1024
    crop_size = 120


class DIV2KTestConfig(DIV2KCommonConfig):
    """Default Dataset config for DIV2K."""

    pass


class DIV2KConfig(object):
    """Default Dataset config for DIV2K."""

    common = DIV2KCommonConfig
    train = DIV2KTrainConfig
    test = DIV2KTestConfig
