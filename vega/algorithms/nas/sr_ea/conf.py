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


class SRPolicyConfig(ConfigSerializable):
    """SR Policy Config."""

    num_sample = 10
    num_mutate = 10

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_SRPolicyConfig = {"num_sample": {"type": int},
                                "num_mutate": {"type": int}
                                }
        return rules_SRPolicyConfig


class SRConfig(ConfigSerializable):
    """SR Config."""

    codec = 'SRCodec'
    policy = SRPolicyConfig
    objective_keys = ['PSNR', 'flops']

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_SRConfig = {"codec": {"type": str},
                          "policy": {"type": dict},
                          "objective_keys": {"type": (list, str)}
                          }
        return rules_SRConfig

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {
            "policy": cls.policy
        }
