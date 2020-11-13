# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Configs."""
from zeus.common import ConfigSerializable


class BohbPolicyConfig(ConfigSerializable):
    """Bohb Policy Config."""

    total_epochs = 81
    min_epochs = 1
    max_epochs = 81
    num_samples = 40
    config_count = 1
    repeat_times = 1
    eta = 3


class BohbConfig(ConfigSerializable):
    """Bobh Config."""

    policy = BohbPolicyConfig
    objective_keys = 'accuracy'
