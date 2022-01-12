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
