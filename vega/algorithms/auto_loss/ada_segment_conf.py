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


class AdaSegPolicyConfig(ConfigSerializable):
    """AdaSegment Policy Config."""

    config_count = 8
    total_rungs = 200
    loss_num = 10

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_AdaSegPolicyConfig = {"config_count": {"type": int},
                                    "total_rungs": {"type": int},
                                    "params_num": {"type": int}
                                    }
        return rules_AdaSegPolicyConfig


class AdaSegConfig(ConfigSerializable):
    """AdaSegment Config."""

    policy = AdaSegPolicyConfig
    objective_keys = 'mAP'

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
