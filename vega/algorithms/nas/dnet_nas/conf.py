# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
