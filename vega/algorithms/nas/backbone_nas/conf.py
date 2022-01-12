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


class BackboneNasPolicyConfig(ConfigSerializable):
    """BackboneNas Policy Config."""

    random_ratio = 0.2
    num_mutate = 10

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_BackboneNasPolicyConfig = {"random_ratio": {"type": float},
                                         "num_mutate": {"type": int}
                                         }
        return rules_BackboneNasPolicyConfig


class BackboneNasRangeConfig(ConfigSerializable):
    """BackboneNas Range Config."""

    max_sample = 100
    min_sample = 10

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_BackboneNasRangeConfig = {"max_sample": {"type": int},
                                        "min_sample": {"type": int}
                                        }
        return rules_BackboneNasRangeConfig


class BackboneNasConfig(ConfigSerializable):
    """BackboneNas Config."""

    codec = 'BackboneNasCodec'
    policy = BackboneNasPolicyConfig
    range = BackboneNasRangeConfig
    pareto = ParetoFrontConfig
    objective_keys = 'accuracy'

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_BackboneNasConfig = {"codec": {"type": str},
                                   "policy": {"type": dict},
                                   "range": {"type": dict},
                                   "pareto": {"type": dict},
                                   "objective_keys": {"type": (list, str)}
                                   }
        return rules_BackboneNasConfig

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {
            "policy": cls.policy,
            "range": cls.range,
            "pareto": cls.pareto
        }
