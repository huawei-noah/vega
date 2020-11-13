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
from zeus.common import ConfigSerializable


class ImagenetCommonConfig(BaseConfig):
    """Default Dataset config for Imagenet."""

    n_class = 1000
    num_workers = 8
    batch_size = 64
    num_parallel_batches = 8
    num_parallel_calls = 64
    fp16 = False
    image_size = 224


class ImagenetTrainConfig(ImagenetCommonConfig):
    """Default Dataset config for Imagenet."""

    shuffle = True
    transforms = [dict(type='RandomResizedCrop', size=224),
                  dict(type='RandomHorizontalFlip'),
                  dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    num_images = 1281167


class ImagenetValConfig(ImagenetCommonConfig):
    """Default Dataset config for Imagenet."""

    transforms = [dict(type='Resize', size=256),
                  dict(type='CenterCrop', size=224),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    num_images = 50000


class ImagenetTestConfig(ImagenetCommonConfig):
    """Default Dataset config for Imagenet."""

    transforms = [dict(type='Resize', size=256),
                  dict(type='CenterCrop', size=224),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    num_images = 50000


class ImagenetConfig(ConfigSerializable):
    """Default Dataset config for Imagenet."""

    common = ImagenetCommonConfig
    train = ImagenetTrainConfig
    val = ImagenetValConfig
    test = ImagenetTestConfig
