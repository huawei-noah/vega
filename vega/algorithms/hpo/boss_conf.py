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


class BossPolicyConfig(ConfigSerializable):
    """Boss Policy Config."""

    # TODO: validation >=3
    total_epochs = 81
    max_epochs = 81
    num_samples = 40
    config_count = 1
    repeat_times = 2

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_BossPolicyConfig = {"total_epochs": {"type": int},
                                  "max_epochs": {"type": int},
                                  "num_samples": {"type": int},
                                  "config_count": {"type": int},
                                  "repeat_times": {"type": int}
                                  }
        return rules_BossPolicyConfig


class BossConfig(ConfigSerializable):
    """Boss Config."""

    policy = BossPolicyConfig
    objective_keys = 'accuracy'

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_BossConfig = {"policy": {"type": dict},
                            "objective_keys": {"type": (list, str)}
                            }
        return rules_BossConfig

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {
            "policy": cls.policy
        }
