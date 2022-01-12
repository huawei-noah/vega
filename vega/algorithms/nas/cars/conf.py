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

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_CARSPolicyConfig = {"momentum": {"type": float},
                                  "weight_decay": {"type": float},
                                  "parallel": {"type": bool},
                                  "warmup": {"type": int},
                                  "sample_num": {"type": int},
                                  "select_method": {"type": str},
                                  "nsga_method": {"type": str},
                                  "pareto_model_num": {"type": int},
                                  "arch_optim": {"type": dict},
                                  "criterion": {"type": dict}
                                  }
        return rules_CARSPolicyConfig


class CARSConfig(ConfigSerializable):
    """CARS Config."""

    codec = 'DartsCodec'
    policy = CARSPolicyConfig
    objective_keys = 'accuracy'

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_CARSConfig = {"codec": {"type": str},
                            "policy": {"type": dict},
                            "objective_keys": {"type": (list, str)}
                            }
        return rules_CARSConfig

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {
            "policy": cls.policy
        }
