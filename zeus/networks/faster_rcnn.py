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


@ClassFactory.register(ClassType.NETWORK)
class FasterRCNN(Module):
    """Create ResNet Network."""

    def __init__(self, num_classes, backbone='SerialBackbone', neck='TorchFPN', network_name='torchvision_FasterRCNN',
                 weight_file=None, **kwargs):
        """Create layers.

        :param num_class: number of class
        :type num_class: int
        """
        super(FasterRCNN, self).__init__()
        self.weight_file = weight_file
        backbone_cls = self.define_props('backbone', backbone, dtype=ClassType.NETWORK)
        backbone_cls.freeze()
        if getattr(backbone_cls, 'out_channels') and 'in_channels' not in neck:
            neck_in_channel = backbone_cls.out_channels
            params = {"in_channels": neck_in_channel}
            neck_cls = self.define_props('neck', neck, dtype=ClassType.NETWORK, params=params)
        else:
            neck_cls = self.define_props('neck', neck, dtype=ClassType.NETWORK)
        backbone_neck = Sequential(backbone_cls, neck_cls)
        backbone_neck.freeze()
        self.model = ClassFactory.get_cls(ClassType.NETWORK, network_name)(backbone_neck, num_classes, **kwargs)

    def call(self, inputs, targets=None):
        """Call inputs."""
        return self.model(inputs, targets)

    def load_state_dict(self, state_dict=None, strict=None):
        """Remove backbone."""
        self.model.load_state_dict(state_dict, strict or False)
