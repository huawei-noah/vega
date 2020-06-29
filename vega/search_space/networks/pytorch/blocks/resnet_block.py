# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined ResNet Blocks."""
import math
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """Generate 3x3 convolution layer."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """Generate 1x1 convolution layer."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LambdaLayer(nn.Module):
    """Layer with lambda function.

    :param lambd: lambda function
    """

    def __init__(self, lambd):
        """Init LambdaLayer."""
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        """Forward function of this layer."""
        return self.lambd(x)


class BasicBlock(nn.Module):
    """BasicBlock for ResNet.

    :param in_planes: input channel.
    :type in_planes: int
    :param out_planes: output channel.
    :type out_planes: int
    :param stride: stride, default is 1.
    :type stride: int
    """

    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        """Init BasicBlock."""
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv_block = nn.Sequential(
            conv3x3(in_planes, out_planes, stride=stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            conv3x3(out_planes, out_planes),
            nn.BatchNorm2d(out_planes),
        )
        self.downsample = None
        if stride != 1 or in_planes != out_planes * self.expansion:
            # For ResNet pruned model
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * self.expansion))

    def forward(self, x):
        """Forward function of BasicBlock."""
        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)
        # forward pass through convolutional block:
        out = self.conv_block(x)
        out += residual
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    """BottleneckBlock for ResNet.

    :param in_planes: input channel.
    :type in_planes: int
    :param out_planes: output channel.
    :type out_planes: int
    :param stride: stride, default is 1.
    :type stride: int
    """

    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        """Init BottleneckBlock."""
        super(BottleneckBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv_block = nn.Sequential(
            conv1x1(in_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            conv3x3(out_planes, out_planes, stride=stride, groups=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            conv1x1(out_planes, out_planes * self.expansion),
            nn.BatchNorm2d(out_planes * self.expansion),
        )
        self.downsample = None
        if stride != 1 or in_planes != out_planes * self.expansion:
            # For ResNet pruned model
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * self.expansion))

    def forward(self, x):
        """Forward function."""
        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)
        # forward pass through convolutional block:
        out = self.conv_block(x)
        out += residual
        out = self.relu(out)
        return out


class SmallInputInitialBlock(nn.Module):
    """SmallInputInitialBlock for ResNet.

    :param in_planes: input channel.
    :type in_planes: int
    """

    def __init__(self, init_planes):
        """Init SmallInputInitialBlock."""
        super().__init__()
        self._module = nn.Sequential(
            conv3x3(3, init_planes, stride=1),
            nn.BatchNorm2d(init_planes),
            nn.ReLU(),
        )

    def forward(self, x):
        """Forward function."""
        return self._module(x)


class InitialBlock(nn.Module):
    """InitialBlock for ResNet.

    :param in_planes: input channel.
    :type in_planes: int
    """

    def __init__(self, init_planes):
        """Init InitialBlock."""
        super().__init__()
        self._module = nn.Sequential(
            nn.Conv2d(3, init_planes, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_planes),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        """Forward function."""
        return self._module(x)


class PruneBasicBlock(nn.Module):
    """Basic block class in prune resnet.

    :param in_planes: input channel number
    :type in_planes: int
    :param planes: output channel number
    :type planes: int
    :param inner_plane: middle layer channel number
    :type inner_plane: int
    :param stride: stride in convolution operation
    :type stride: int
    :param option: shortcut type
    :type option: 'A' or 'B'
    """

    expansion = 1

    def __init__(self, in_planes, planes, inner_plane, stride=1, option='B'):
        """Init PruneBasicBlock."""
        super(PruneBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, inner_plane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner_plane)
        self.conv2 = nn.Conv2d(inner_plane, planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                # For ResNet pruned model, use option B
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        """Forward function of BasicBlock."""
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        out += F.relu(self.shortcut(x))
        return out
