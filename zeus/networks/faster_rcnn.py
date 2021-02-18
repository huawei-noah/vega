# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is FasterRCNN network."""
from zeus.common import ClassFactory, ClassType
from zeus.modules.module import Module
from zeus.modules.connections import Sequential
from torchvision.models import detection


@ClassFactory.register(ClassType.NETWORK)
class FasterRCNN(Module):
    """Create ResNet Network."""

    def __init__(self, num_classes, backbone='ResNetDet', neck='FPN', **kwargs):
        """Create layers.

        :param num_class: number of class
        :type num_class: int
        """
        super(FasterRCNN, self).__init__()
        self.backbone = self.define_props('backbone', backbone, dtype=ClassType.NETWORK, params=dict(depth=18))
        self.neck = self.define_props('neck', neck, dtype=ClassType.NETWORK)
        backbone_neck = Sequential(self.backbone, self.neck)
        self.model = detection.FasterRCNN(backbone_neck, num_classes, **kwargs)

    def call(self, inputs, targets=None):
        """Call inputs."""
        inputs = list(image.cuda() for image in inputs)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        return self.model(inputs, targets)
