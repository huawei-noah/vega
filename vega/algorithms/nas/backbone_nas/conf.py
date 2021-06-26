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
