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


class AshaPolicyConfig(ConfigSerializable):
    """Asha Policy Config."""

    total_epochs = 50
    max_epochs = 81
    config_count = 1
    num_samples = 9


class AshaConfig(ConfigSerializable):
    """Asha Config."""

    policy = AshaPolicyConfig
    objective_keys = 'accuracy'
