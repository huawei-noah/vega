# -*- coding:utf-8 -*-

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

"""Fuse Operator."""
import logging
import copy
import vega
from vega.modules.operators.ops import Identity

if vega.is_torch_backend():
    import torch
    from torch.nn.utils.fusion import fuse_conv_bn_weights


def fuse(model, weights_file=None):
    """Fuse Conv and BN."""
    if not vega.is_torch_backend() or model.__class__.__name__ != 'DagNetwork':
        return model
    logging.info("Start operator fusion.")
    for name, node in model.module_map.items():
        module = node.module
        if isinstance(node.module, torch.nn.Conv2d):
            next_nodes = node.child_nodes
            if next_nodes and isinstance(next_nodes[0].module, torch.nn.BatchNorm2d):
                node.module = _fuse_conv_bn(module, next_nodes[0].module)
                next_nodes[0].module = Identity()
    if weights_file:
        _save_model(model, weights_file)
    return model


def _fuse_conv_bn(conv, bn):
    fused_conv = copy.deepcopy(conv)
    fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
        fused_conv.weight, fused_conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    return fused_conv


def _save_model(model, weights_file):
    if vega.is_torch_backend():
        torch.save(model.state_dict(), weights_file)
