# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Configs."""


class BoPolicyConfig(object):
    """Bo Policy Config."""

    total_epochs = 10
    epochs_per_iter = 1
    warmup_count = 5
    alg_name = 'SMAC'


class BoConfig(object):
    """Bo Config."""

    policy = BoPolicyConfig
    objective_keys = 'accuracy'
