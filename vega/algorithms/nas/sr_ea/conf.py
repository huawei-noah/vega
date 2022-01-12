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
