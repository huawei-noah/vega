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
"""This is Ghost Module for blocks."""
from vega.common import ClassType, ClassFactory
from vega.modules.module import Module
from vega.modules.operators import ops


class Bottleneck(Module):
    """Bottleneck class."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = ops.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = ops.BatchNorm2d(planes)
        self.conv2 = ops.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = ops.BatchNorm2d(planes)
        self.conv3 = ops.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = ops.BatchNorm2d(planes * 4)
        self.relu = ops.Relu(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward x."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


@ClassFactory.register(ClassType.NETWORK)
class GhostModule(Module):
    """Ghost Module."""

    def __init__(self, inplanes, planes, blocks, stride=1, cheap_ratio=0.5):
        super(GhostModule, self).__init__()
        from torch.nn import Sequential
        block = Bottleneck
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = Sequential(
                ops.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                ops.BatchNorm2d(planes * block.expansion),
            )

        self.blocks = blocks
        self.base = block(inplanes, planes, stride, downsample)
        self.end = block(planes * block.expansion, planes, 1)

        if blocks > 2:
            self.c_base_half = planes * block.expansion // 2
            inplanes = planes * block.expansion // 2
            cheap_planes = int(planes * cheap_ratio)
            self.cheap_planes = cheap_planes
            raw_planes = planes - cheap_planes
            self.merge = Sequential(
                ops.AdaptiveAvgPool2d(1),
                ops.Conv2d(raw_planes * block.expansion * blocks, cheap_planes * block.expansion, kernel_size=1,
                           stride=1, bias=False),
                ops.BatchNorm2d(cheap_planes * block.expansion),
            )
            self.cheap = Sequential(
                ops.Conv2d(planes * block.expansion, cheap_planes * block.expansion,
                           kernel_size=1, stride=1, padding=0, bias=False),
                ops.BatchNorm2d(cheap_planes * block.expansion),
            )
            self.cheap_relu = ops.Relu(inplace=True)

            layers = []

            inplanes = raw_planes * block.expansion
            layers.append(
                ops.Conv2d(planes * block.expansion, inplanes, kernel_size=1, stride=1, padding=0, bias=False))  #

            for i in range(1, blocks - 1):
                layers.append(block(inplanes, raw_planes))
            self.layers = Sequential(*layers)

    def forward(self, input):
        """Forward x."""
        x0 = self.base(input)

        if self.blocks > 2:
            m_list = [x0]
            x = x0
            for n, l in enumerate(self.layers):
                x = l(x)
                if n != 0:
                    m_list.append(x)
            m = ops.concat(m_list, 1)

            m = self.merge(m)
            if self.cheap_planes > 0:
                c = x0
                c = self.cheap_relu(self.cheap(c) + m)
                x = ops.concat((x, c), 1)
            x = self.end(x)
        else:
            x = self.end(x0)
        return x

    def to_desc(self, recursion=True):
        """Convert to desc."""
        return {"type": "GhostModule", "inplanes": self.inplanes, "planes": self.planes, "blocks": self.blocks,
                "stride": self.stride}
