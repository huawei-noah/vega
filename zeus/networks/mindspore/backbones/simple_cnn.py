# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Simple CNN network."""

import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from zeus.common import ClassType, ClassFactory


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Conv layer weight initial."""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="same")


def fc_with_initialize(input_channels, out_channels):
    """Fc layer weight initial."""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """Weight initial."""
    return TruncatedNormal(0.02)


@ClassFactory.register(ClassType.NETWORK)
class SimpleCnn(nn.Cell):
    """Lenet network structure."""

    # define the operator required
    def __init__(self, num_class=None, blocks=None, channels=None, *args, **kwargs):
        super(SimpleCnn, self).__init__()
        self.num_class = num_class
        self.blocks = blocks
        self.channels = channels
        self.conv1 = conv(3, self.channels, 3)
        self.conv2 = conv(self.channels, self.channels, 3)
        self.conv3 = conv(self.channels, self.channels * 2, 3)
        self.fc1 = fc_with_initialize(self.channels * 2 * 4 * 4, 128)
        self.fc2 = fc_with_initialize(128, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        """Construct."""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
