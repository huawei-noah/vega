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


class ESRPolicyConfig(ConfigSerializable):
    """ESR Policy Config."""

    num_generation = 2
    num_individual = 4
    num_elitism = 2
    mutation_rate = 0.05


class ESRRangeConfig(ConfigSerializable):
    """ESR Range Config."""

    node_num = 20
    min_active = 16
    max_params = 1020000
    min_params = 1010000


class ESRConfig(ConfigSerializable):
    """ESR Config."""

    codec = 'ESRCodec'
    policy = ESRPolicyConfig
    range = ESRRangeConfig
    objective_keys = 'SRMetric'
