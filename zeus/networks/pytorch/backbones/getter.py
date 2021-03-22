# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ResNetVariant for Detection."""
from zeus.common import ClassType, ClassFactory
from zeus.modules.connections.connections import MultiOutputGetter


@ClassFactory.register(ClassType.NETWORK)
class BackboneGetter(MultiOutputGetter):
    """Backbone Getter form torchvision ResNet."""

    def __init__(self, backbone_name, layer_names=None, **kwargs):
        backbone = ClassFactory.get_cls(ClassType.NETWORK, backbone_name)
        backbone = backbone(**kwargs) if kwargs else backbone()
        if hasattr(backbone, "layers_name"):
            layer_names = backbone.layers_name()
        layer_names = layer_names or ['layer1', 'layer2', 'layer3', 'layer4']
        super(BackboneGetter, self).__init__(backbone, layer_names)
