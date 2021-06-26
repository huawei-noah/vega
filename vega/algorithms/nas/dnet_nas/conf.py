# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Configs."""

from vega.core.search_algs import ParetoFrontConfig
from vega.common import ConfigSerializable


class DblockNasRangeConfig(ConfigSerializable):
    """DnetNas Range Config."""

    max_sample = 100
    min_sample = 10


class DblockNasConfig(ConfigSerializable):
    """DblockNas Config."""

    codec = 'DblockNasCodec'
    range = DblockNasRangeConfig
    pareto = ParetoFrontConfig
    objective_keys = 'accuracy'


class DnetNasPolicyConfig(ConfigSerializable):
    """DnetNas Policy Config."""

    random_ratio = 0.2
    num_mutate = 10


class DnetNasRangeConfig(ConfigSerializable):
    """DnetNas Range Config."""

    max_sample = 100
    min_sample = 10


class DnetNasConfig(ConfigSerializable):
    """DnetNas Config."""

    codec = 'DnetNasCodec'
    policy = DnetNasPolicyConfig
    range = DnetNasRangeConfig
    pareto = ParetoFrontConfig
    objective_keys = 'accuracy'
