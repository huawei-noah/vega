# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""This is Network for SerialNet."""
from vega.common import ClassFactory, ClassType
from vega.modules.operators import ops
from vega.modules.module import Module
from vega.modules.connections import Sequential, OutDictSequential
from vega.modules.blocks.head import LinearClassificationHead


@ClassFactory.register(ClassType.NETWORK)
class SerialClassificationNet(Module):
    """SerialClassificationNet."""

    def __init__(self, code='111-2111-211111-211', num_classes=1000, block='BottleneckBlock', in_channels=64,
                 weight_file=None):
        """Init SerialClassificationNet."""
        super(SerialClassificationNet, self).__init__()
        self.backbone = SerialBackbone(code, block, in_channels, weight_file, out_layers=-1)
        self.head = LinearClassificationHead(self.out_channels, num_classes)


@ClassFactory.register(ClassType.NETWORK)
class SerialBackbone(Module):
    """Serial Net for spnas."""

    def __init__(self, code='111-2111-211111-211', block='BottleneckBlock', in_channels=64, weight_file=None,
                 out_layers=None):
        """Init SerialBackbone."""
        super(SerialBackbone, self).__init__()
        self.inplanes = in_channels
        self.planes = self.inplanes
        self.weight_file = weight_file
        self.channels = [3]
        self.code = code.split('-')
        self.block = ClassFactory.get_cls(ClassType.NETWORK, block)
        self._make_stem_layer()
        self.layers = Sequential() if out_layers == -1 else OutDictSequential()
        self.make_cells()

    @property
    def out_channels(self):
        """Output Channel for Module."""
        return self.layers.out_channels

    def load_state_dict(self, state_dict=None, strict=None):
        """Load and freeze backbone state."""
        if isinstance(self.layers, Sequential):
            return super().load_state_dict(state_dict, strict or False)
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
        state_dict = {k.replace('body.', ''): v for k, v in state_dict.items()}
        not_swap_keys = super().load_state_dict(state_dict, strict or False)
        if not_swap_keys:
            need_freeze_layers = [name for name, parameter in self.named_parameters() if name not in not_swap_keys]
            self.freeze(need_freeze_layers)
        else:
            self.freeze(['layers'])

    def _make_stem_layer(self):
        """Make stem layer."""
        self.conv1 = ops.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = ops.BatchNorm2d(self.inplanes)
        self.relu = ops.Relu(inplace=True)
        self.maxpool = ops.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def make_cells(self):
        """Make ResNet Cell."""
        for i, code in enumerate(self.code):
            layer, planes = self.make_layers(self.block, self.inplanes, self.planes, code=code)
            self.channels.append(planes)
            self.inplanes = planes
            self.layers.append(layer)
            self.planes = self.planes * 2

    def make_layers(self, block, inplanes, planes, code=None):
        """Make ResNet layers."""
        strides = list(map(int, code))
        layers = []
        layers.append(block(inplanes, planes, stride=strides[0]))
        inplanes = planes * block.expansion
        for stride in strides[1:]:
            layers.append(block(inplanes, planes, stride=stride))
            inplanes = planes * block.expansion
        return Sequential(*layers), inplanes
