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
