# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
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
