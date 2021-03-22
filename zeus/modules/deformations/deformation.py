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
from zeus.modules.getters import GraphGetter
from zeus.modules.operators import ops
from zeus.modules.connections import Add


@ClassFactory.register(ClassType.NETWORK)
class Deformation(Module):
    """Get output layer by layer names and connect into a OrderDict."""

    def __init__(self, desc, from_graph=None, weight_file=None):
        super(Deformation, self).__init__()
        self._desc = {"props": deepcopy(desc.get('props')) or {}}
        if from_graph:
            self.model = GraphGetter(desc, weight_file).model
        else:
            self.model = ModelZoo().get_model(desc, weight_file)
        self._apply_names()
        self.get_search_space()
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

    def to_desc(self, recursion=True):
        """Convert to model desc."""
        return dict(self.model.to_desc(), **self._desc)

    def state_dict(self):
        """Get state dict."""
        return self.model.state_dict()

    def get_search_space(self):
        """Get Search Space from model."""
        search_space = DeformationSearchSpace()
        search_space._traversal(self.model)
        self.search_space = search_space.search_space
        self.conditions = search_space.conditions


class DeformationSearchSpace(object):
    """Get SearchSpace."""

    def __init__(self):
        self.pre_conv = None
        self.search_space = OrderedDict()
        self.conditions = deque()

    def _traversal(self, module):
        for module in module.children():
            if isinstance(module, Add):
                self._traversal_add_connections(module)
            elif isinstance(module, ops.Conv2d):
                self.search_space[module.name + '/out_channels'] = module.out_channels
                if self.pre_conv:
                    self.add_condition(module.name + '/in_channels', self.pre_conv.name + '/out_channels')
                self.pre_conv = module
            elif isinstance(module, ops.BatchNorm2d):
                self.add_condition(module.name + '/num_features', self.pre_conv.name + '/out_channels')
            elif isinstance(module, ops.Linear):
                self.add_condition(module.name + '/in_features', self.pre_conv.name + '/out_channels')
            else:
                self._traversal(module)
        return self

    def _traversal_add_connections(self, module):
        last_convs = []
        for child in module.children():
            add_convs = [conv for name, conv in child.named_modules() if isinstance(conv, ops.Conv2d)]
            if add_convs:
                last_convs.append(add_convs[-1])
        tmp_pre_conv = self.pre_conv
        for child in module.children():
            self.pre_conv = tmp_pre_conv
            self._traversal(child)
        if len(last_convs) > 1:
            self.pre_conv = last_convs[0]
            last_convs = last_convs[1:]
        for conv in last_convs:
            self.add_condition(conv.name + '/out_channels', self.pre_conv.name + '/out_channels')
        # The out_channels value of the jump node is the same as that of the previous nodes
        # remove from the search space.
        for last_conv in last_convs:
            if self.pre_conv == last_conv:
                continue
            self.search_space.popitem(last_conv.name + '/out_channels')
            for k, v in [(k, v) for k, v in self.conditions if v == last_conv.name + '/out_channels']:
                self.add_condition(k, self.pre_conv.name + '/out_channels')
        self.pre_conv = last_convs[0]

    def add_condition(self, name, value):
        """Add condition."""
        self.conditions.append((name, value))
