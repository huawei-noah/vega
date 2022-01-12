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


class BohbPolicyConfig(ConfigSerializable):
    """Bohb Policy Config."""

    total_epochs = 81
    min_epochs = 1
    max_epochs = 81
    num_samples = 40
    config_count = 1
    repeat_times = 1
    eta = 3

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_BohbPolicyConfig = {"total_epochs": {"type": int},
                                  "min_epochs": {"type": int},
                                  "max_epochs": {"type": int},
                                  "num_samples": {"type": int},
                                  "config_count": {"type": int},
                                  "repeat_times": {"type": int},
                                  "eta": {"type": int}
                                  }
        return rules_BohbPolicyConfig


class BohbConfig(ConfigSerializable):
    """Bobh Config."""

    policy = BohbPolicyConfig
    objective_keys = 'accuracy'
    random_samples = None    # 32
    prob_crossover = 0.6
    prob_mutatation = 0.2
    tuner = "RF"    # TPE | GP | RF

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
