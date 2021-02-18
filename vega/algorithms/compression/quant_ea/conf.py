# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Configs."""

from vega.core.search_algs import EAConfig
from zeus.common import ConfigSerializable


class QuantPolicyConfig(EAConfig):
    """Quant Policy Config."""

    length = 40
    num_generation = 50
    num_individual = 16
    random_models = 32


class QuantConfig(ConfigSerializable):
    """Quant Config."""

    codec = 'QuantCodec'
    policy = QuantPolicyConfig
    objective_keys = ['accuracy', 'flops']
