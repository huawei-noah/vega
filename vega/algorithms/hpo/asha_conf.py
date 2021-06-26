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


class AshaPolicyConfig(ConfigSerializable):
    """Asha Policy Config."""

    total_epochs = 50
    max_epochs = 81
    config_count = 1
    num_samples = 9
    eta = 3

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_AshaPolicyConfig = {"total_epochs": {"type": int},
                                  "max_epochs": {"type": int},
                                  "config_count": {"type": int},
                                  "num_samples": {"type": int}
                                  }
        return rules_AshaPolicyConfig


class AshaConfig(ConfigSerializable):
    """Asha Config."""

    policy = AshaPolicyConfig
    objective_keys = 'accuracy'

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_AshaConfig = {"policy": {"type": dict},
                            "objective_keys": {"type": (list, str)}
                            }
        return rules_AshaConfig

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {
            "policy": cls.policy
        }
