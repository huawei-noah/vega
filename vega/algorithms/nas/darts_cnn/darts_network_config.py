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
from vega.core.common import Config


class DartsNetworkTemplateConfig(object):
    """Darts network template config."""

    cifar10 = Config(os.path.join(os.path.dirname(__file__), "darts_cifar10.json"))
    imagenet = Config(os.path.join(os.path.dirname(__file__), "darts_imagenet.json"))
