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
