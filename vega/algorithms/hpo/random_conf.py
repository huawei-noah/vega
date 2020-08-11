# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Configs."""


class RandomPolicyConfig(object):
    """Random Policy Config."""

    total_epochs = 10
    epochs_per_iter = 1


class RandomConfig(object):
    """Random Config."""

    policy = RandomPolicyConfig
    objective_keys = 'accuracy'
