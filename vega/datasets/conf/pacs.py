# -*- coding=utf-8 -*-

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
"""Default configs."""

from vega.common import ConfigSerializable
from .base import BaseConfig


class PacsCommonConfig(BaseConfig):
    """Default Dataset config for PacsCommon."""
    transforms = [
        dict(type='Resize', size=225),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    batch_size = 10
    data_path = None
    split_path = None
    targetdomain = None
    task = None

    @classmethod
    def rules(cls):
        """Return rules for checking."""

        rules_PacsCommonConfig = {"transforms": {"type": list},
                                  "data_path": {"type": str},
                                  "split_path": {"type": str},
                                  "targetdomain": {"type": str},
                                  "batch_size": {"type": int},
                                 }
        return rules_PacsCommonConfig


class PacsConfig(ConfigSerializable):
    """Default Dataset config for Pacs."""

    common = PacsCommonConfig
    train = PacsCommonConfig
    val = PacsCommonConfig
    test = PacsCommonConfig

    @classmethod
    def rules(cls):
        """Return rules for checking."""

        rules_Pacs = {'common': {"type": dict},
                      'train': {"type": dict},
                      'val': {"type": dict},
                      'test': {"type": dict}
                       }
        return rules_Pacs

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {'common': cls.common,
                'train': cls.train,
                'val': cls.val,
                'test': cls.test
                }
