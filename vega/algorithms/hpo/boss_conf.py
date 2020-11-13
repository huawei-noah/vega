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


class BossPolicyConfig(ConfigSerializable):
    """Boss Policy Config."""

    # TODO: validation >=3
    total_epochs = 81
    max_epochs = 81
    num_samples = 40
    config_count = 1
    repeat_times = 2


class BossConfig(ConfigSerializable):
    """Boss Config."""

    policy = BossPolicyConfig
    objective_keys = 'accuracy'
