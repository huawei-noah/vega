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


class PBTPolicyConfig(ConfigSerializable):
    """PBT Policy Config."""

    config_count = 16
    each_epochs = 3
    total_rungs = 200

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_PBTPolicyConfig = {"config_count": {"type": int},
                                 "each_epochs": {"type": int},
                                 "total_rungs": {"type": int}
                                 }
        return rules_PBTPolicyConfig


class PBTConfig(ConfigSerializable):
    """PBT Config."""

    policy = PBTPolicyConfig
    objective_keys = 'accuracy'

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
