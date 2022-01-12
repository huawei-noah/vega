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
