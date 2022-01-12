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


class BayesPolicyConfig(ConfigSerializable):
    """Bohb Policy Config."""

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_BohbPolicyConfig = {"num_samples": {"type": int},
                                  "warmup_count": {"type": int},
                                  "prob_mutatation": {"type": float},
                                  "prob_crossover": {"type": float},
                                  "tuner": {"type": str},
                                  }
        return rules_BohbPolicyConfig


class BayesConfig(ConfigSerializable):
    """Bayes Config."""

    policy = BayesPolicyConfig
    objective_keys = 'accuracy'
    num_samples = 32
    warmup_count = 16
    prob_mutatation = 0.2
    prob_crossover = 0.6
    tuner = "RF"  # TPE | GP | RF

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_BoConfig = {"policy": {"type": dict},
                          "objective_keys": {"type": (list, str)}
                          }
        return rules_BoConfig

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {
            "policy": cls.policy
        }
