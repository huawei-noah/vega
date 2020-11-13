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


class AdelaideConfig(ConfigSerializable):
    """AdelaideMutate Config."""

    codec = 'AdelaideCodec'
    max_sample = 10
    pareto_front_file = "{local_base_path}/output/random/pareto_front.csv"
    random_file = "{local_base_path}/output/random/random.csv"
    objective_keys = ['IoUMetric', 'flops']
