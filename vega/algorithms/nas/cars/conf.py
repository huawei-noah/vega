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


class CARSPolicyConfig(EAConfig):
    """CARS Policy Config."""

    momentum = 0.9
    weight_decay = 3.0e-4
    parallel = False
    expand = 1.0
    warmup = 50
    sample_num = 1
    select_method = 'uniform'  # pareto
    nsga_method = 'cars_nsga'
    pareto_model_num = 4
    arch_optim = dict(type='Adam', lr=3.0e-4, betas=[0.5, 0.999], weight_decay=1.0e-3)
    criterion = dict(type='CrossEntropyLoss')


class CARSConfig(ConfigSerializable):
    """CARS Config."""

    codec = 'DartsCodec'
    policy = CARSPolicyConfig
    objective_keys = 'accuracy'
