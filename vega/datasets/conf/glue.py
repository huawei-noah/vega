# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Dataset configs."""
from vega.datasets.conf.base import BaseConfig
from vega.common import ConfigSerializable


class GlueCommonConfig(BaseConfig):
    """Default Dataset config for Bert."""

    batch_size = 32
    task_name = None
    max_seq_length = 128
    vocab_file = None
    do_lower_case = True
    pregenerated = False
    transforms = dict(type='ToTensorAll')


class GlueTrainConfig(GlueCommonConfig):
    """Default Dataset config for Bert train."""

    pass


class GlueValConfig(GlueCommonConfig):
    """Default Dataset config for Bert val."""

    pass


class GlueTestConfig(GlueCommonConfig):
    """Default Dataset config for Bert val."""

    pass


class GlueConfig(ConfigSerializable):
    """Default Dataset config for Glue."""

    common = GlueCommonConfig
    train = GlueTrainConfig
    val = GlueValConfig
    test = GlueTestConfig
