# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Simple CNN network."""

from zeus.modules.operators import ops
from zeus.modules.connections import ModuleList, Sequential
from zeus.modules.module import Module
from zeus.common.config import Config
from zeus.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.NETWORK)
class SimpleCnn(Module):
    """Simple CNN network."""

    def __init__(self, **desc):
        """Initialize."""
        super(SimpleCnn, self).__init__()
        desc = Config(**desc)
        self.num_class = desc.num_class
        self.fp16 = desc.get('fp16', False)
        self.channels = desc.channels
        self.conv1 = ops.Conv2d(3, 32, padding=1, kernel_size=3)
        self.pool1 = ops.MaxPool2d(2, stride=2)
        self.blocks = self._blocks(self.channels, desc.blocks)
        self.pool2 = ops.MaxPool2d(2, stride=2)
        self.conv2 = ops.Conv2d(self.channels, 64, padding=1, kernel_size=3)
        self.global_conv = ops.Conv2d(64, 64, kernel_size=8)
        self.view = ops.View()
        self.fc = ops.Linear(64, self.num_class)

    def _blocks(self, out_channels, desc_blocks):
        blocks = ModuleList()
        in_channels = 32
        for i in range(desc_blocks):
            blocks.append(Sequential(
                ops.Conv2d(in_channels, out_channels, padding=1, kernel_size=3),
                ops.BatchNorm2d(out_channels),
                ops.Relu(inplace=True),
            ))
            in_channels = out_channels
        return blocks

    def call(self, x):
        """Call."""
        x = self.pool1(self.conv1(x))
        for block in self.blocks:
            x = block(x)
        x = self.global_conv(self.conv2(self.pool2(x)))
        x = self.fc(self.view(x))
        return x
