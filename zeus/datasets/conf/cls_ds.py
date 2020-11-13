# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""

from zeus.common import ConfigSerializable


class ClassificationDatasetCommonConfig(ConfigSerializable):
    """Default Optim Config."""

    data_path = None
    batch_size = 1
    shuffle = False
    drop_last = True
    n_class = None
    train_portion = 1.0
    n_images = None
    cached = True
    transforms = []
    num_workers = 1
    distributed = False
    pin_memory = False


class ClassificationDatasetTraineConfig(ClassificationDatasetCommonConfig):
    """Default Cifar10 config."""

    shuffle = True


class ClassificationDatasetValConfig(ClassificationDatasetCommonConfig):
    """Default Cifar10 config."""

    shuffle = False


class ClassificationDatasetTestConfig(ClassificationDatasetCommonConfig):
    """Default Cifar10 config."""

    shuffle = False


class ClassificationDatasetConfig(ConfigSerializable):
    """Default Dataset config for Cifar10."""

    common = ClassificationDatasetCommonConfig
    train = ClassificationDatasetTraineConfig
    val = ClassificationDatasetValConfig
    test = ClassificationDatasetTestConfig
