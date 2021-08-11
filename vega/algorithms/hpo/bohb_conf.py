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
