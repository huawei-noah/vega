# -*- coding: utf-8 -*-

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

"""Simple CNN network."""

from vega.modules.operators import ops
from vega.modules.connections import ModuleList, Sequential
from vega.modules.module import Module
from vega.common.config import Config
from vega.common import ClassType, ClassFactory


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
        self.global_conv = ops.Conv2d(64, 64, kernel_size=8, padding=0)
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
