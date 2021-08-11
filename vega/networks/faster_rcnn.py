# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is FasterRCNN network."""
from vega.common import ClassFactory, ClassType
from vega.modules.connections import Sequential
from vega.modules.module import Module
from torchvision.models.detection import FasterRCNN


@ClassFactory.register(ClassType.NETWORK)
class FasterRCNN(FasterRCNN, Module):
    """Create ResNet Network."""

    def __init__(self, num_classes=81, backbone='ResNetBackbone', neck='FPN', **kwargs):
        """Create layers.

        :param num_class: number of class
        :type num_class: int
        """
        backbone_cls = ClassFactory.get_instance(ClassType.NETWORK, backbone)
        neck_cls = ClassFactory.get_instance(ClassType.NETWORK, neck, in_channels=backbone_cls.out_channels)
        backbone_neck = Sequential()
        backbone_neck.append(backbone_cls, 'body')
        backbone_neck.append(neck_cls, 'fpn')
        super(FasterRCNN, self).__init__(backbone_neck, num_classes, **kwargs)
