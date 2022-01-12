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

from vega.datasets.conf.base import BaseConfig
from vega.common import ConfigSerializable


class MrpcCommonConfig(BaseConfig):
    """Default Dataset config for Bert."""

    max_seq_length = 128
    vocab_file = None
    do_lower_case = True
    transforms = dict(type='ToTensorAll')


class MrpcTrainConfig(MrpcCommonConfig):
    """Default Dataset config for Bert train."""

    pass


class MrpcValConfig(MrpcCommonConfig):
    """Default Dataset config for Bert val."""

    pass


class MrpcTestConfig(MrpcCommonConfig):
    """Default Dataset config for Bert val."""

    pass


class MrpcConfig(ConfigSerializable):
    """Default Dataset config for Coco."""

    common = MrpcCommonConfig
    train = MrpcTrainConfig
    val = MrpcValConfig
    test = MrpcTestConfig
