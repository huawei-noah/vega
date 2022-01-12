# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ops for NAGO."""
import torch
from torch import nn


class depthwise_separable_conv_general(nn.Module):
    """Depthwise seperable convolution operation."""

    def __init__(self, nin, nout, stride, kernel_size=3, padding=None):
        """Initialize depthwise_separable_conv_general."""
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        """Implement forward."""
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Triplet_unit(nn.Module):
    """Node operation unit in the bottom-level graph."""

    def __init__(self, inplanes, outplanes, dropout_p=0, stride=1, kernel_size=3):
        """Initialize Triplet_unit."""
        super(Triplet_unit, self).__init__()
        self.relu = nn.ReLU()
        self.conv = depthwise_separable_conv_general(inplanes, outplanes, stride, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(outplanes)
        self.dropout_p = dropout_p
        if dropout_p > 0:
            self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """Implement forward."""
        out = self.relu(x)
        out = self.conv(out)
        out = self.bn(out)
        if self.dropout_p > 0:
            out = self.dropout(out)
        return out


class PassThrough(nn.Module):
    """Class PassThrough."""

    def forward(self, x):
        """Forward method."""
        return x


class BoundedScalarMultiply(nn.Module):
    """Class BoundedScalarMultiply."""

    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.ones(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward method."""
        return self.sigmoid(self.mean) * x
