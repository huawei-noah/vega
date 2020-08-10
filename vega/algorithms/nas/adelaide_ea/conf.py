# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Configs."""


class AdelaideConfig(object):
    """AdelaideMutate Config."""

    codec = 'AdelaideCodec'
    max_sample = 10
    pareto_front_file = "{local_base_path}/output/random1/pareto_front.csv"
    random_file = "{local_base_path}/output/random1/random.csv"
    objective_keys = ['IoUMetric', 'gflops']
