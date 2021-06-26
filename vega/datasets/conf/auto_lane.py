# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""

from .base import BaseConfig
from vega.common import ConfigSerializable


class AutoLaneCommonConfig(BaseConfig):
    """Default Dataset config for Curve Lane."""

    network_input_width = 512
    network_input_height = 288
    gt_len = 145
    gt_num = 576
    batch_size = 24
    num_workers = 4
    shuffle = True
    random_sample = True
    transforms = [dict(type='ToTensor'),
                  dict(type='Normalize',
                       mean=[0.49139968, 0.48215827, 0.44653124],
                       std=[0.24703233, 0.24348505, 0.26158768])]

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Common_AutoLane = {"network_input_width": {"type": int},
                                 "network_input_height": {"type": int},
                                 "gt_len": {"type": int},
                                 "gt_num": {"type": int},
                                 "batch_size": {"type": int},
                                 "num_workers": {"type": int},
                                 "shuffle": {"type": bool},
                                 "random_sample": {"type": bool},
                                 "transforms": {"type": list}
                                 }
        return rules_Common_AutoLane


class AutoLaneTrainConfig(AutoLaneCommonConfig):
    """Default Dataset config for Curve Lane."""

    pass


class AutoLaneValConfig(AutoLaneCommonConfig):
    """Default Dataset config for Curve Lane."""

    pass


class AutoLaneTestConfig(AutoLaneCommonConfig):
    """Default Dataset config for Curve Lane."""

    pass


class AutoLaneConfig(ConfigSerializable):
    """Default Dataset config for Curve Lane."""

    common = AutoLaneCommonConfig
    train = AutoLaneTrainConfig
    val = AutoLaneValConfig
    test = AutoLaneTestConfig

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_AutoLane = {"common": {"type": dict},
                          "train": {"type": dict},
                          "val": {"type": dict},
                          "test": {"type": dict}
                          }
        return rules_AutoLane

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {'common': cls.common,
                'train': cls.train,
                'val': cls.val,
                'test': cls.test
                }
