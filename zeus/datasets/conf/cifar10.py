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


class Cifar10CommonConfig(BaseConfig):
    """Default Optim Config."""

    n_class = 10
    batch_size = 256
    num_workers = 8
    train_portion = 1.0
    num_parallel_batches = 64
    fp16 = False


class Cifar10TrainConfig(Cifar10CommonConfig):
    """Default Cifar10 config."""

    transforms = [
        dict(type='RandomCrop', size=32, padding=4),
        dict(type='RandomHorizontalFlip'),
        dict(type='ToTensor'),
        # rgb_mean = np.mean(train_data, axis=(0, 1, 2))/255
        # rgb_std = np.std(train_data, axis=(0, 1, 2))/255
        dict(type='Normalize', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])]
    padding = 8
    num_images = 50000


class Cifar10ValConfig(Cifar10CommonConfig):
    """Default Cifar10 config."""

    transforms = [
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])]
    num_images = 10000
    num_images_train = 50000


class Cifar10TestConfig(Cifar10CommonConfig):
    """Default Cifar10 config."""

    transforms = [
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])]
    num_images = 10000


class Cifar10Config(ConfigSerializable):
    """Default Dataset config for Cifar10."""

    common = Cifar10CommonConfig
    train = Cifar10TrainConfig
    val = Cifar10ValConfig
    test = Cifar10TestConfig
