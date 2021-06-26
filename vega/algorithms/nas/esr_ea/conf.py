# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Configs."""
from vega.common import ConfigSerializable


class ESRPolicyConfig(ConfigSerializable):
    """ESR Policy Config."""

    num_generation = 2
    num_individual = 4
    num_elitism = 2
    mutation_rate = 0.05

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_ESRPolicyConfig = {"num_generation": {"type": int},
                                 "num_individual": {"type": int},
                                 "num_elitism": {"type": int},
                                 "mutation_rate": {"type": float}
                                 }
        return rules_ESRPolicyConfig


class ESRRangeConfig(ConfigSerializable):
    """ESR Range Config."""

    node_num = 20
    min_active = 16
    max_params = 1020000
    min_params = 1010000

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_ESRRangeConfig = {"node_num": {"type": int},
                                "min_active": {"type": int},
                                "max_params": {"type": int},
                                "min_params": {"type": int}
                                }
        return rules_ESRRangeConfig


class ESRConfig(ConfigSerializable):
    """ESR Config."""

    codec = 'ESRCodec'
    policy = ESRPolicyConfig
    range = ESRRangeConfig
    objective_keys = 'PSNR'

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_ESRConfig = {"codec": {"type": str},
                           "policy": {"type": dict},
                           "range": {"type": dict},
                           "objective_keys": {"type": (list, str)}
                           }
        return rules_ESRConfig

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {
            "policy": cls.policy,
            "range": cls.range
        }
