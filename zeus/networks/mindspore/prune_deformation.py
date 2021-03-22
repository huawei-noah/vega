# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Deformation for Network."""
from copy import deepcopy
from collections import OrderedDict, deque
from zeus.common import ClassType, ClassFactory
from zeus.modules.module import Module
from zeus.model_zoo import ModelZoo
from mindspore import nn
from zeus.modules.operators import PruneConv2DFilter, PruneBatchNormFilter, PruneLinearFilter


@ClassFactory.register(ClassType.NETWORK)
class Deformation(Module):
    """Get output layer by layer names and connect into a OrderDict."""

    def __init__(self, desc, weight_file=None):
        super(Deformation, self).__init__()
        self.search_space = OrderedDict()
        self.conditions = deque()
        self.desc = deepcopy(desc.get('desc')) or desc
        self.model = ModelZoo().get_model(desc, weight_file)
        self.get_search_space(self.model)
        self.decode()
        self.deform()
        self.props.clear()

    def deform(self):
        """Deform Modules."""
        raise NotImplementedError

    def decode(self):
        """Decode Condition and search space."""
        for k, v in self.conditions:
            if self.props.get(v):
                self.props[k] = self.props.get(v)

    def get_search_space(self, model=None, pre_conv=None):
        """Get Search Space from model."""
        if model is None:
            model = self.model
        pre_conv_name = ""
        for name, module in model.cells_and_names():
            # if isinstance(module, nn.SequentialCell):
            #     self.get_search_space(module)
            if isinstance(module, nn.Conv2d):
                self.search_space[name + '.out_channels'] = module.out_channels
                if pre_conv:
                    self.add_condition(name + '.in_channels', pre_conv_name + '.out_channels')
                pre_conv = module
                pre_conv_name = name
            elif "downsample" in name:
                pre_conv = self._get_add_search_space(module, pre_conv)
            elif isinstance(module, nn.BatchNorm2d):
                self.add_condition(name + '.num_features', pre_conv_name + '.out_channels')
            elif isinstance(module, nn.Dense):
                self.add_condition(name + '.in_features', pre_conv_name + '.out_channels')
        return self.search_space

    def add_condition(self, name, value):
        """Add condition."""
        self.conditions.append((name, value))

    def _get_add_search_space(self, module, pre_conv):
        last_convs = []
        for child in module.cells_and_names():
            add_convs = [conv for name, conv in child.name_cells() if isinstance(conv, nn.Conv2d)]
            if add_convs:
                last_convs.append(add_convs[-1])
        # for child in module.children():
        #     self.get_search_space(child, pre_conv)
        if len(last_convs) > 1:
            pre_conv = last_convs[0]
            last_convs = last_convs[1:]
        for conv in last_convs:
            self.add_condition(conv.model_name + '.out_channels', pre_conv.model_name + '.out_channels')
        # The out_channels value of the jump node is the same as that of the previous nodes
        # remove from the search space.
        for last_conv in last_convs:
            self.search_space.popitem(last_conv.model_name + '.out_channels')
            for k, v in [(k, v) for k, v in self.conditions if v == last_conv.model_name + '.out_channels']:
                self.add_condition(k, pre_conv.model_name + '.out_channels')
        return last_convs[0]


@ClassFactory.register(ClassType.NETWORK)
class PruneDeformation(Deformation):
    """Prune any Network."""

    def deform(self):
        """Deform Network."""
        if not self.props or self.desc:
            return
        for name, module in self.model.cells_and_names():
            if isinstance(module, nn.Conv2d):
                PruneConv2DFilter(module, self.props).filter()
            elif isinstance(module, nn.BatchNorm2d):
                PruneBatchNormFilter(module, self.props).filter()
            elif isinstance(module, nn.Dense):
                PruneLinearFilter(module, self.props).filter()
