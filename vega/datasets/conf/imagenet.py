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


class ImagenetCommonConfig(BaseConfig):
    """Default Dataset config for Imagenet."""

    n_class = 1000
    num_workers = 8
    batch_size = 64
    num_parallel_batches = 8
    num_parallel_calls = 64
    fp16 = False
    image_size = 224
    train_portion = 1.0

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_ImagenetCommon = {"n_class": {"type": int},
                                "num_workers": {"type": int},
                                "batch_size": {"type": int},
                                "num_parallel_batches": {"type": int},
                                "num_parallel_calls": {"type": int},
                                "fp16": {"type": int},
                                "image_size": {"type": int}
                                }
        return rules_ImagenetCommon


class ImagenetTrainConfig(ImagenetCommonConfig):
    """Default Dataset config for Imagenet."""

    shuffle = True
    transforms = [dict(type='RandomResizedCrop', size=224),
                  dict(type='RandomHorizontalFlip'),
                  dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    num_images = 1281167

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_ImagenetTrain = {"shuffle": {"type": bool},
                               "transforms": {"type": list},
                               "num_images": {"type": int}
                               }
        return rules_ImagenetTrain


class ImagenetValConfig(ImagenetCommonConfig):
    """Default Dataset config for Imagenet."""

    transforms = [dict(type='Resize', size=256),
                  dict(type='CenterCrop', size=224),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    num_images = 50000

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_ImagenetVal = {"transforms": {"type": list},
                             "num_images": {"type": int}
                             }
        return rules_ImagenetVal


class ImagenetTestConfig(ImagenetCommonConfig):
    """Default Dataset config for Imagenet."""

    transforms = [dict(type='Resize', size=256),
                  dict(type='CenterCrop', size=224),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    num_images = 50000

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_ImagenetTest = {"transforms": {"type": list},
                              "num_images": {"type": int}
                              }
        return rules_ImagenetTest


class ImagenetConfig(ConfigSerializable):
    """Default Dataset config for Imagenet."""

    common = ImagenetCommonConfig
    train = ImagenetTrainConfig
    val = ImagenetValConfig
    test = ImagenetTestConfig

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Imagenet = {"common": {"type": dict},
                          "train": {"type": dict},
                          "val": {"type": dict},
                          "test": {"type": dict}
                          }
        return rules_Imagenet

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {'common': cls.common,
                'train': cls.train,
                'val': cls.val,
                'test': cls.test
                }
