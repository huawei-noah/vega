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


class RandomPolicyConfig(ConfigSerializable):
    """Random Policy Config."""

    num_sample = 10

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_RandomPolicyConfig = {
            "num_sample": {"type": int}
        }
        return rules_RandomPolicyConfig


class RandomConfig(ConfigSerializable):
    """Random Config."""

    policy = RandomPolicyConfig
    objective_keys = 'accuracy'

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_RandomConfig = {"policy": {"type": dict},
                              "objective_keys": {"type": (list, str)}
                              }
        return rules_RandomConfig

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {
            "policy": cls.policy
        }
