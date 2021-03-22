# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""BackboneDeformation for Network."""
import logging
from zeus.common import ClassType, ClassFactory
from .deformation import Deformation
from zeus.modules.operators import ops


@ClassFactory.register(ClassType.NETWORK)
class BackboneDeformation(Deformation):
    """Get output layer by layer names and connect into a OrderDict."""

    def __init__(self, desc, from_graph=False, weight_file=None):
        super(BackboneDeformation, self).__init__(desc, from_graph, weight_file)
        self.is_adaptive_weight = True

    def decode(self):
        """Decode Condition and search space."""
        if not self.props or 'doublechannel' not in self.props or 'downsample' not in self.props:
            return
        logging.info("codec:{}".format(self.props))
        names = list(self.search_space.keys())
        double_channels = self.props.pop('doublechannel')
        for idx, item in enumerate(double_channels):
            if item == 1:
                name = names[idx]
                self.props[name] = self.search_space.get(name) * 2
        down_samples = self.props.pop('downsample')
        for idx, item in enumerate(down_samples):
            if item == 1:
                name = names[idx]
                self.props[name] = self.search_space.get(name) // 2
        for k, v in self.conditions:
            if k.endswith('out_channels') and self.props.get(v) and k not in self.props:
                self.props[k] = self.props.get(v)
        logging.info("Props:{}".format(self.props))

    def deform(self):
        """Deform Backbone Module."""
        if not self.props:
            return
        for name, module in self.model.named_modules():
            if isinstance(module, ops.Conv2d):
                dist_out_channels = self.props.get(module.name + '/out_channels')
                if dist_out_channels:
                    module.out_channels = dist_out_channels
