# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Configs."""
from vega.common import ConfigSerializable


class EAConfig(ConfigSerializable):
    """Base Config for EA."""

    num_individual = 8  # 128
    num_individual_per_iter = 1
    num_generation = 1
    start_ga_epoch = 0  # 50
    ga_interval = 1  # 10
    warmup = 0  # 50
