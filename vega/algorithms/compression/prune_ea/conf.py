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

"""Defined Prune-EA configs."""

from vega.core.search_algs import EAConfig
from vega.common import ConfigSerializable


class PrunePolicyConfig(EAConfig):
    """Prune Policy Config."""

    length = 464
    num_generation = 31
    num_individual = 4
    random_samples = 32

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_PrunePolicyConfig = {"length": {"type": int},
                                   "num_generation": {"type": int},
                                   "num_individual": {"type": int},
                                   "random_samples": {"type": int}
                                   }
        return rules_PrunePolicyConfig


class PruneConfig(ConfigSerializable):
    """Prune Config."""

    codec = 'PruneCodec'
    policy = PrunePolicyConfig
    objective_keys = ['accuracy', 'flops']

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_PruneConfig = {"codec": {"type": str},
                             "policy": {"type": dict},
                             "objective_keys": {"type": (list, str)}
                             }
        return rules_PruneConfig

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {
            "policy": cls.policy
        }
