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


class REDSCommonConfig(BaseConfig):
    """Common config of REDS."""

    dataroot_gt = None
    dataroot_lq = None
    num_frame = 5


class REDSTrainConfig(REDSCommonConfig):
    """Train config of REDS."""

    meta_info_file = None
    val_partition = 'REDS4'
    gt_size = 256
    interval_list = [1]
    random_reverse = False
    use_flip = True
    use_rot = True
    use_shuffle = True
    num_worker = 3
    batch_size = 4
    repeat_ratio = 200
    prefetch_mode = None


class REDSValConfig(REDSCommonConfig):
    """Valid config of REDS."""

    meta_info_file = None
    cache_data = False
    padding = 'reflection_circle'
    batch_size = 1
    num_workers = 3


class REDSTestConfig(REDSCommonConfig):
    """Test config of REDS."""

    pass


class REDSConfig(ConfigSerializable):
    """Default Dataset config for REDS."""

    common = REDSCommonConfig
    train = REDSTrainConfig
    val = REDSValConfig
    test = REDSTestConfig
