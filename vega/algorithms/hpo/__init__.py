# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Lazy import hpo algorithms."""

from zeus.common.class_factory import ClassFactory


ClassFactory.lazy_register("vega.algorithms.hpo", {
    "asha_hpo": ["AshaHpo"],
    "bo_hpo": ["BoHpo"],
    "bohb_hpo": ["BohbHpo"],
    "boss_hpo": ["BossHpo"],
    "random_hpo": ["RandomSearch"],
    "random_pareto_hpo": ["RandomParetoHpo"],
    "evolution_search": ["EvolutionAlgorithm"],
})
