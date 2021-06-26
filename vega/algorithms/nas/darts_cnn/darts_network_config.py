# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Contains Default and User configuration."""
import os
from vega.common import Config
from vega.common import ConfigSerializable


class DartsNetworkTemplateConfig(ConfigSerializable):
    """Darts network template config."""

    cifar10 = Config(os.path.join(os.path.dirname(__file__), "darts_cifar10.json"))
    cifar100 = Config(os.path.join(os.path.dirname(__file__), "darts_cifar100.json"))
    imagenet = Config(os.path.join(os.path.dirname(__file__), "darts_imagenet.json"))
