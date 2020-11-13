# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""

from zeus.datasets.conf.base import BaseConfig
from zeus.common import ConfigSerializable


class CocoCommonConfig(BaseConfig):
    """Default Dataset config for Coco."""

    data_root = None
    num_classes = 81
    img_prefix = '2017'
    ann_prefix = 'instances'
    transforms = [dict(type='PolysToMaskTransform'), dict(type='PILToTensor')]


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
