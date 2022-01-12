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


class PBAPolicyConfig(ConfigSerializable):
    """PBA Policy Config."""

    total_epochs = 81
    config_count = 16
    each_epochs = 3
    total_rungs = 200

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_PBAPolicyConfig = {"total_epochs": {"type": int},
                                 "config_count": {"type": int},
                                 "each_epochs": {"type": int},
                                 "total_rungs": {"type": int}
                                 }
        return rules_PBAPolicyConfig


class PBAConfig(ConfigSerializable):
    """PBA Config."""

    policy = PBAPolicyConfig
    objective_keys = 'accuracy'
    transformers = dict(Cutout=True,
                        Rotate=True,
                        Translate_X=True,
                        Translate_Y=True,
                        Brightness=True,
                        Color=True,
                        Invert=True,
                        Sharpness=True,
                        Posterize=True,
                        Shear_X=True,
                        Solarize=True,
                        Shear_Y=True,
                        Equalize=True,
                        AutoContrast=True,
                        Contras=True)

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_PBAConfig = {"policy": {"type": dict},
                           "objective_keys": {"type": (list, str)},
                           "transformers": {"type": dict}
                           }
        return rules_PBAConfig

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {
            "policy": cls.policy
        }
