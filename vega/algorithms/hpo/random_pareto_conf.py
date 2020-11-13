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
from zeus.common import ConfigSerializable


class RandomParetoPolicyConfig(ConfigSerializable):
    """Random Pareto Policy Config."""

    total_epochs = 10
    max_epochs = 1


class RandomParetoConfig(ConfigSerializable):
    """Random Pareto Config."""

    policy = RandomParetoPolicyConfig
    pareto = ParetoFrontConfig
    objective_keys = 'accuracy'
