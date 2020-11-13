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


class DIV2KTrainConfig(DIV2KCommonConfig):
    """Default Dataset config for DIV2K."""

    load_size = 1024
    crop_size = 120


class DIV2KTestConfig(DIV2KCommonConfig):
    """Default Dataset config for DIV2K."""

    pass


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
