# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for blocks."""
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.fine_grained_space.fine_grained_space import FineGrainedSpace
from vega.search_space.fine_grained_space.operators import op, conv1X1, conv3x3
from vega.search_space.fine_grained_space.conditions import Add


@ClassFactory.register(ClassType.SEARCH_SPACE)
class ShortCut(FineGrainedSpace):
    """Create Shortcut SearchSpace."""

    def __init__(self, inchannel, outchannel, expansion, stride=1):
        """Create ShortCut layer.

        :param inchannel: input channel.
        :type inchannel: int
        :param outchannel: output channel.
        :type outchannel: int
        :param expansion: expansion
        :type expansion: int
        :param stride: the number to jump, default 1
        :type stride: int
        """
        super(ShortCut, self).__init__(inchannel, outchannel, expansion, stride)
        if stride != 1 or inchannel != outchannel * expansion:
            self.conv1 = conv1X1(inchannel=inchannel, outchannel=outchannel * expansion, stride=stride)
            self.batch = op.BatchNorm2d(num_features=outchannel * expansion)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class BottleConv(FineGrainedSpace):
    """Create BottleConv Searchspace."""

    def __init__(self, inchannel, outchannel, expansion, groups, base_width, stride=1):
        """Create BottleConv layer.

        :param inchannel: input channel.
        :type inchannel: int
        :param outchannel: output channel.
        :type outchannel: int
        :param expansion: expansion
        :type expansion: int
        :param stride: the number to jump, default 1
        :type stride: int
        """
        super(BottleConv, self).__init__(inchannel, outchannel, groups, base_width, stride=1)
        outchannel = int(outchannel * (base_width / 64.)) * groups
        self.conv1 = conv1X1(inchannel=inchannel, outchannel=outchannel)
        self.batch1 = op.BatchNorm2d(num_features=outchannel)
        self.conv2 = conv3x3(inchannel=outchannel, outchannel=outchannel, groups=groups, stride=stride)
        self.batch2 = op.BatchNorm2d(num_features=outchannel)
        self.conv3 = conv1X1(inchannel=outchannel, outchannel=outchannel * expansion)
        self.batch3 = op.BatchNorm2d(num_features=outchannel * expansion)
        self.relu = op.ReLU()


@ClassFactory.register(ClassType.SEARCH_SPACE)
class BasicConv(FineGrainedSpace):
    """Create BasicConv Searchspace."""

    def __init__(self, inchannel, outchannel, groups=1, base_width=64, stride=1, inner_plane=None):
        """Create BasicConv layer.

        :param inchannel: input channel.
        :type inchannel: int
        :param outchannel: output channel.
        :type outchannel: int
        :param stride: the number to jump, default 1
        :type stride: int
        """
        super(BasicConv, self).__init__(inchannel, outchannel, groups, base_width, stride, inner_plane)
        if inner_plane is None:
            if groups != 1 or base_width != 64:
                raise ValueError("BasicBlock only supports groups=1 and base_width=64")
            self.conv = conv3x3(inchannel=inchannel, outchannel=outchannel, stride=stride)
            self.batch = op.BatchNorm2d(num_features=outchannel)
            self.relu = op.ReLU(inplace=True)
            self.conv2 = conv3x3(inchannel=outchannel, outchannel=outchannel)
            self.batch2 = op.BatchNorm2d(num_features=outchannel)
        else:
            self.conv = conv3x3(inchannel=inchannel, outchannel=inner_plane, stride=stride)
            self.batch = op.BatchNorm2d(num_features=inner_plane)
            self.relu = op.ReLU(inplace=True)
            self.conv2 = conv3x3(inchannel=inner_plane, outchannel=outchannel)
            self.batch2 = op.BatchNorm2d(num_features=outchannel)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class SmallInputInitialBlock(FineGrainedSpace):
    """Create SmallInputInitialBlock SearchSpace."""

    def __init__(self, init_plane):
        """Create SmallInputInitialBlock layer.

        :param init_plane: input channel.
        :type init_plane: int
        """
        super(SmallInputInitialBlock, self).__init__(init_plane)
        self.conv = conv3x3(inchannel=3, outchannel=init_plane, stride=1)
        self.batch = op.BatchNorm2d(num_features=init_plane)
        self.relu = op.ReLU()


@ClassFactory.register(ClassType.SEARCH_SPACE)
class InitialBlock(FineGrainedSpace):
    """Create InitialBlock SearchSpace."""

    def __init__(self, init_plane):
        """Create InitialBlock layer.

        :param init_plane: input channel.
        :type init_plane: int
        """
        super(InitialBlock, self).__init__(init_plane)
        self.conv = op.Conv2d(in_channels=3, out_channels=init_plane, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch = op.BatchNorm2d(num_features=init_plane)
        self.relu = op.ReLU()
        self.maxpool2d = op.MaxPool2d(kernel_size=3, stride=2, padding=1)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class BasicBlock(FineGrainedSpace):
    """Create BasicBlock SearchSpace."""

    expansion = 1

    def __init__(self, inchannel, outchannel, groups=1, base_width=64, stride=1, inner_plane=None):
        """Create BasicBlock layers.

        :param inchannel: input channel.
        :type inchannel: int
        :param outchannel: output channel.
        :type outchannel: int
        :param stride: the number to jump, default 1
        :type stride: int
        """
        super(BasicBlock, self).__init__(inchannel, outchannel, groups, base_width, stride, inner_plane)
        base_conv = BasicConv(inchannel=inchannel, outchannel=outchannel, stride=stride,
                              groups=groups, base_width=base_width, inner_plane=inner_plane)
        shortcut = ShortCut(inchannel=inchannel, outchannel=outchannel, expansion=self.expansion,
                            stride=stride)
        self.block = Add(base_conv, shortcut)
        self.relu = op.ReLU()


@ClassFactory.register(ClassType.SEARCH_SPACE)
class BottleneckBlock(FineGrainedSpace):
    """Create BottleneckBlock SearchSpace."""

    expansion = 4

    def __init__(self, inchannel, outchannel, groups, base_width, stride=1):
        """Create BottleneckBlock layers.

        :param inchannel: input channel.
        :type inchannel: int
        :param outchannel: output channel.
        :type outchannel: int
        :param stride: the number to jump, default 1
        :type stride: int
        """
        super(BottleneckBlock, self).__init__(inchannel, outchannel, groups, base_width, stride)
        bottle_conv = BottleConv(inchannel=inchannel, outchannel=outchannel, expansion=self.expansion,
                                 stride=stride, groups=groups, base_width=base_width)
        shortcut = ShortCut(inchannel=inchannel, outchannel=outchannel, expansion=self.expansion, stride=stride)
        self.block = Add(bottle_conv, shortcut)
        self.relu = op.ReLU()
