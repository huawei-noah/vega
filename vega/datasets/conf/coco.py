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

from vega.datasets.conf.base import BaseConfig
from vega.common import ConfigSerializable


class CocoCommonConfig(BaseConfig):
    """Default Dataset config for Coco."""

    data_root = None
    num_classes = 81
    img_prefix = '2017'
    ann_prefix = 'instances'
    test_size = 50
    transforms = [dict(type='PolysToMaskTransform'), dict(type='PILToTensorAll')]

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_CocoCommon = {"data_root": {"type": str},
                            "num_classes": {"type": int},
                            "img_prefix": {"type": (int, str)},
                            "ann_prefix": {"type": str},
                            "transforms": {"type": list}
                            }
        return rules_CocoCommon


class CocoTrainConfig(CocoCommonConfig):
    """Default Dataset config for Coco train."""

    shuffle = True


class CocoValConfig(CocoCommonConfig):
    """Default Dataset config for Coco val."""

    pass


class CocoTestConfig(CocoCommonConfig):
    """Default Dataset config for Coco val."""

    pass


class CocoConfig(ConfigSerializable):
    """Default Dataset config for Coco."""

    common = CocoCommonConfig
    train = CocoTrainConfig
    val = CocoValConfig
    test = CocoTestConfig

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Coco = {"common": {"type": dict},
                      "train": {"type": dict},
                      "val": {"type": dict},
                      "test": {"type": dict}
                      }
        return rules_Coco

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {'common': cls.common,
                'train': cls.train,
                'val': cls.val,
                'test': cls.test
                }
