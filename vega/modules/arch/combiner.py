# -*- coding: utf-8 -*-

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

"""ConnectionsArchParamsCombiner."""
from collections import deque
from vega.modules.operators import ops
from vega.modules.connections import Add
from vega.modules.module import Module


def is_depth_wise_conv(module):
    """Determine Conv2d."""
    if hasattr(module, "groups"):
        return module.groups != 1 and module.in_channels == module.out_channels
    elif hasattr(module, "group"):
        return module.group != 1 and module.in_channels == module.out_channels


class ConnectionsArchParamsCombiner(object):
    """Get ConnectionsArchParamsCombiner."""

    def __init__(self):
        self.pre_conv = None
        self.arch_type = None
        self.search_space = []
        self.conditions = deque()
        self.forbidden = []

    def get_search_space_by_arch_type(self, module, arch_type):
        """Get Search Space."""
        self.arch_type = arch_type
        self._traversal(module)
        return self.search_space

    def combine(self, module):
        """Decode modules arch params."""
        self._traversal(module)
        module.set_arch_params({k: v for k, v in module._arch_params.items() if k not in self.forbidden})
        if module._arch_params_type == 'Prune':
            for k, v in self.conditions:
                if module._arch_params.get(v):
                    module._arch_params[k] = module._arch_params.get(v)
        return self.search_space

    def _traversal(self, module):
        """Traversal search space and conditions."""
        if isinstance(module, Add):
            self._traversal_add_connections(module)
        elif isinstance(module, ops.Conv2d):
            if self.pre_conv:
                self.add_condition(module.name + '.in_channels', self.pre_conv.name + '.out_channels')
                if is_depth_wise_conv(module):
                    self.add_condition(module.name + '.out_channels', module.name + '.in_channels')
            self.pre_conv = module
        elif isinstance(module, ops.BatchNorm2d):
            self.add_condition(module.name + '.num_features', self.pre_conv.name + '.out_channels')
        elif isinstance(module, ops.Linear):
            self.add_condition(module.name + '.in_features', self.pre_conv.name + '.out_channels')
        elif module.__class__.__name__ == "Reshape":
            self.add_condition(module.name + '.shape', self.pre_conv.name + '.out_channels')
        elif isinstance(module, Module):
            for child in module.children():
                self._traversal(child)

    def _traversal_add_connections(self, module):
        last_convs = []
        last_bns = []
        add_bns = []
        for child in module.children():
            if isinstance(child, ops.Conv2d):
                add_convs = [child]
            elif isinstance(child, ops.Identity):
                continue
            else:
                add_convs = [conv for name, conv in child.named_modules() if isinstance(conv, ops.Conv2d)]
                add_bns = [bn for name, bn in child.named_modules() if isinstance(bn, ops.BatchNorm2d)]
            if add_convs:
                last_convs.append(add_convs[-1])
            if add_bns:
                last_bns.append(add_bns[-1])
        tmp_pre_conv = self.pre_conv
        for child in module.children():
            self.pre_conv = tmp_pre_conv
            self._traversal(child)
        if len(last_convs) > 1:
            self.pre_conv = last_convs[0]
            last_convs = last_convs[1:]
        else:
            self.pre_conv = tmp_pre_conv
        for conv in last_convs:
            self.add_condition(conv.name + '.out_channels', self.pre_conv.name + '.out_channels')
        # The out_channels value of the jump node is the same as that of the previous nodes
        # remove from the search space.
        if len(last_convs) == 1:
            self.add_forbidden(last_convs[0].name + '.out_channels')
            self.add_condition(last_convs[0].name + '.out_channels', self.pre_conv.name + '.out_channels')
            if len(last_bns) > 0:
                self.add_condition(last_bns[-1].name + '.num_features', self.pre_conv.name + '.out_channels')
        else:
            for last_conv in last_convs:
                if self.pre_conv == last_conv:
                    continue
                self.add_forbidden(last_conv.name + '.out_channels')
                self.add_condition(last_convs[0].name + '.out_channels', self.pre_conv.name + '.out_channels')
                for k, v in [(k, v) for k, v in self.conditions if v == last_conv.name + '.out_channels']:
                    self.add_condition(k, self.pre_conv.name + '.out_channels')
        if len(last_convs) > 0:
            self.pre_conv = last_convs[0]

    def add_condition(self, name, value):
        """Add condition."""
        self.conditions.append((name, value))

    def add_forbidden(self, name):
        """Add condition."""
        self.forbidden.append(name)
