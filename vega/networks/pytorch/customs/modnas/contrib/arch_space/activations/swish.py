# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Swish activation functions."""
import torch.nn as nn
import torch.nn.functional as F
from modnas.registry.arch_space import register


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation function."""

    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """Return module output."""
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class HardSwish(nn.Module):
    """Hard Swish activation function."""

    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """Return module output."""
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Swish(nn.Module):
    """Swish activation function."""

    def forward(self, x):
        """Return module output."""
        return x * F.sigmoid(x)


register(HardSigmoid)
register(HardSwish)
register(Swish)
