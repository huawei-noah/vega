# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for network."""
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module
from .resnet_general import ResNetGeneral
from vega.modules.operators.ops import Linear, AdaptiveAvgPool2d, View


@ClassFactory.register(ClassType.NETWORK)
class ResNet(Module):
    """Create ResNet SearchSpace."""

    def __init__(self, depth=18, base_channel=64, out_plane=None, stages=4, num_class=10, small_input=True,
                 doublechannel=None, downsample=None):
        """Create layers.

        :param num_reps: number of layers
        :type num_reqs: int
        :param items: channel and stride of every layer
        :type items: dict
        :param num_class: number of class
        :type num_class: int
        """
        super(ResNet, self).__init__()
        self.backbone = ResNetGeneral(small_input, base_channel, depth, stages, doublechannel, downsample)
        self.adaptiveAvgPool2d = AdaptiveAvgPool2d(output_size=(1, 1))
        self.view = View()
        out_plane = out_plane or self.backbone.out_channels
        self.head = Linear(in_features=out_plane, out_features=num_class)
