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
from vega.common import ConfigSerializable


class QuantPolicyConfig(EAConfig):
    """Quant Policy Config."""

    length = 40
    num_generation = 50
    num_individual = 16
    random_samples = 32

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_QuantPolicyConfig = {"length": {"type": int},
                                   "num_generation": {"type": int},
                                   "num_individual": {"type": int},
                                   "random_samples": {"type": int}
                                   }
        return rules_QuantPolicyConfig


class QuantConfig(ConfigSerializable):
    """Quant Config."""

    codec = 'QuantCodec'
    policy = QuantPolicyConfig
    objective_keys = ['accuracy', 'flops']

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_QuantConfig = {"codec": {"type": str},
                             "policy": {"type": dict},
                             "objective_keys": {"type": (list, str)}
                             }
        return rules_QuantConfig

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {
            "policy": cls.policy
        }
