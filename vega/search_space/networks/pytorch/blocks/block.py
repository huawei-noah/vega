# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Block."""
import torch
import torch.nn as nn
from vega.search_space.networks.pytorch.network import Network


class Block(Network):
    """Block base class."""

    def __init__(self):
        """Init Block."""
        super().__init__()
        self.block = None

    def forward(self, x):
        """Forward function."""
        return self.block(x)
