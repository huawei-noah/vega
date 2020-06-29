# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Utils for network register."""
from enum import Enum


class NetTypes(Enum):
    """NetTypes."""

    BLOCK = 1
    CUSTOM = 2
    BACKBONE = 3
    HEAD = 7
    LOSS = 8
    SUPER_NETWORK = 10
    ESRBODY = 11
    OPS = 12
    UTIL = 14
    TORCH_VISION_MODEL = 15
    Operator = 16


NetTypesMap = {
    "block": NetTypes.BLOCK,
    "custom": NetTypes.CUSTOM,
    "backbone": NetTypes.BACKBONE,
    "head": NetTypes.HEAD,
    "loss": NetTypes.LOSS,
    "super_network": NetTypes.SUPER_NETWORK,
    "esrbody": NetTypes.ESRBODY,
    "ops": NetTypes.OPS,
    "util": NetTypes.UTIL,
    "torch_vision_model": NetTypes.TORCH_VISION_MODEL,
    'operator': NetTypes.Operator,
}

NetworkRegistry = dict()
NetworkNameRegistry = dict()

for type in NetTypes:
    NetworkRegistry[type] = dict()
