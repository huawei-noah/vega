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


class AvazuCommonConfig(BaseConfig):
    """Default Dataset config for Avazu."""

    batch_size = 2000
    block_size = 2000000
    train_num_of_files = 17
    test_num_of_files = 5
    train_size = 32343173
    test_size = 8085794
    pos_train_samples = 0
    pos_test_samples = 0
    neg_train_samples = 0
    neg_test_samples = 0
    train_pos_ratio = 0
    test_pos_ratio = 0
    initialized = True
    max_length = 24
    num_of_feats = 645195
    feat_names = ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                  'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16',
                  'C17', 'C18', 'C19', 'C20', 'C21', 'mday', 'hour', 'wday']
    feat_sizes = [7, 7, 3135, 3487, 24, 4002, 252, 28, 101449, 523672, 5925, 5, 4, 2417, 8, 9, 426, 4, 67, 166, 60, 10,
                  24, 7]
    random_sample = False
    shuffle_block = False


class AvazuTrainConfig(AvazuCommonConfig):
    """Default Dataset config for Avazu."""

    pass


class AvazuValConfig(AvazuCommonConfig):
    """Default Dataset config for Avazu."""

    pass


class AvazuTestConfig(AvazuCommonConfig):
    """Default Dataset config for Avazu."""

    pass


class AvazuConfig(ConfigSerializable):
    """Default Dataset config for Avazu."""

    common = AvazuCommonConfig
    train = AvazuTrainConfig
    val = AvazuValConfig
    test = AvazuTestConfig
