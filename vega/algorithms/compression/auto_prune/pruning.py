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

"""Prune model."""
import copy

import vega
from vega.common import ClassType, ClassFactory

if vega.is_torch_backend():
    import torch


def is_conv2d(module):
    """Determine Conv2d."""
    # depth-wise convolution not in pruned search space.
    return isinstance(module, torch.nn.Conv2d) and not is_depth_wise_conv(module)


def is_depth_wise_conv(module):
    """Determine Conv2d."""
    if hasattr(module, "groups"):
        return module.groups != 1 and module.in_channels == module.out_channels
    elif hasattr(module, "group"):
        return module.group != 1 and module.in_channels == module.out_channels


def mask_conv2d_out_channels(module, value):
    """Mask out channels of Conv2d."""
    if not value:
        return
    out_channels_idx = [idx for idx, value in enumerate(value) if value == 0]
    if not out_channels_idx:
        return
    module.weight.data[out_channels_idx, :, :, :] = 0


def mask_conv2d_in_channels(module, value):
    """Mask in channels of conv2d."""
    if not value:
        return
    in_channels_idx = [idx for idx, value in enumerate(value) if value == 0]
    if not in_channels_idx:
        return
    module.weight.data[:, in_channels_idx, :, :] = 0


def mask_conv2d(module, c_in, c_out):
    """Mask conv2d."""
    if not isinstance(module, torch.nn.Conv2d):
        return
    mask_conv2d_in_channels(module, c_in)
    mask_conv2d_out_channels(module, c_out)


def mask_linear(module, value):
    """Mask linear."""
    if not isinstance(module, torch.nn.Linear) or not value or sum(value) == len(value):
        return
    idx_in = [idx for idx, value in enumerate(value) if value == 0]
    if not idx_in:
        return
    module.weight.data[:, idx_in] = 0


def mask_batch_norm(module, value):
    """Prune Batch Norm."""
    if not isinstance(module, torch.nn.BatchNorm2d) or not value:
        return
    idx = [idx for idx, value in enumerate(value) if value == 0]
    if not idx:
        return
    weights = {**module._parameters, **module._buffers}
    if 'num_batches_tracked' in weights:
        weights.pop('num_batches_tracked')
    for name, weight in weights.items():
        if name == 'running_mean':
            module.running_mean.data[idx] = 0
        elif name == 'running_var':
            module.running_var.data[idx] = 0
        elif name == 'weight':
            module.weight.data[idx] = 0
        elif name == 'bias':
            module.bias.data[idx] = 0


def prune_conv2d_out_channels(module, value):
    """Prune out channels of Conv2d."""
    if not value:
        return
    module.out_channels = sum(value)
    if sum(value) != module.out_channels:
        raise ValueError("Outchannel is wrong.")
    out_channels_idx = [idx for idx, value in enumerate(value) if value == 1]
    for name, weight in module._parameters.items():
        if weight is None:
            continue
        if name == 'weight':
            module.weight.data = weight[out_channels_idx, :, :, :]
        elif name == 'bias':
            module.bias.data = weight[out_channels_idx]


def prune_conv2d_in_channels(module, value):
    """Prune in channels of conv2d."""
    if not value:
        return
    module.in_channels = sum(value)
    if sum(value) != module.in_channels:
        raise ValueError("Inchannel is wrong.")
    in_channels_idx = [idx for idx, value in enumerate(value) if value == 1]
    for name, weight in module._parameters.items():
        if weight is None or name != 'weight':
            continue
        if hasattr(module, "groups") and module.groups != 1:
            # group and depth-wise convolution
            # todo: not working on BINARY_CODE mode, mask code must be divisible by weight
            module.groups = module.in_channels // weight.shape[1]
        else:
            prune_weight = weight[:, in_channels_idx, :, :]
            module.weight.data = prune_weight


def prune_conv2d(module, c_in, c_out):
    """prune conv2d."""
    if not isinstance(module, torch.nn.Conv2d):
        return
    prune_conv2d_in_channels(module, c_in)
    prune_conv2d_out_channels(module, c_out)


def prune_linear(module, value):
    """Prune linear."""
    if not isinstance(module, torch.nn.Linear) or not value or sum(value) == len(value):
        return
    if module.in_features == len(value):
        module.in_features = sum(value)
    else:
        module.in_features = module.in_features // len(value) * sum(value)
    if sum(value) == module.in_features:
        idx_in = [idx for idx, value in enumerate(value) if value == 1]
    else:
        idx_in = [idx for idx, value in enumerate([1] * module.in_features)]
    module.weight.data = module.weight.data[:, idx_in]


def prune_batch_norm(module, value):
    """Prune Batch Norm."""
    if not isinstance(module, torch.nn.BatchNorm2d) or not value:
        return
    module.num_features = sum(value)
    if sum(value) != module.num_features:
        raise ValueError("Features is wrong.")
    idx = [idx for idx, value in enumerate(value) if value == 1]
    weights = {**module._parameters, **module._buffers}
    if 'num_batches_tracked' in weights:
        weights.pop('num_batches_tracked')
    for name, weight in weights.items():
        prune_weight = weight[idx]
        if name == 'running_mean':
            module.running_mean.data = prune_weight
        elif name == 'running_var':
            module.running_var.data = prune_weight
        elif name == 'weight':
            module.weight.data = prune_weight
        elif name == 'bias':
            module.bias.data = prune_weight


def prune_dag_model(model):
    """Prune Dag model."""
    for name, node in model.named_nodes():
        prune_conv2d(node.module, node.c_in, node.c_out)
        prune_batch_norm(node.module, node.c_out)
        prune_linear(node.module, node.c_in)
        if node.module_type == 'torch_tensor_view':
            if node.c_in and len(node.c_in) != sum(node.c_in) and node.saved_args and len(node.saved_args) > 1:
                node.saved_args = tuple([node.saved_args[0], node.saved_args[1] // len(node.c_in) * sum(node.c_in)])
    return model


def prune_model(model, dag):
    """Prune Dag model."""
    for name, module in model.named_modules():
        node = dag.module_map.get(name)
        if not node:
            continue
        prune_conv2d(module, node.c_in, node.c_out)
        prune_batch_norm(module, node.c_out)
        prune_linear(module, node.c_in)
    return model


def mask_model(model, dag):
    """Mask model."""
    for name, module in model.named_modules():
        node = dag.module_map.get(name)
        if not node:
            continue
        mask_conv2d(module, node.c_in, node.c_out)
        mask_batch_norm(module, node.c_out)
        mask_linear(module, node.c_in)
    return model


def named_pruned_modules(model):
    """Get call pruned modules."""
    for name, module in model.named_modules():
        if is_conv2d(module):
            yield name, module


def decode(model, desc, strategy=None):
    """Decode desc into mask code."""
    mask_code_desc = {}
    trans = MaskCodeTransformer(strategy)
    for name, rate in desc.items():
        node_name = '.'.join(name.split('.')[:-1])
        arch_type = name.split('.')[-1]
        if node_name not in model.module_map:
            continue
        node_channels = model.module_map[node_name].module.out_channels
        if arch_type == 'prune_d_rate':
            select_idx = round(node_channels * rate / 100 / 16) * 16
            select_idx = select_idx if select_idx > 16 else node_channels
        else:
            select_idx = node_channels * rate // 100
        mask_code_desc[node_name + '.out_channels'] = trans(model, select_idx, node_channels, node_name)
    return mask_code_desc


class MaskCodeTransformer(object):
    """Transform mask code."""

    def __init__(self, strategy=''):
        self.strategy = strategy

    def __call__(self, model, select_idx, node_channels, node_name):
        """Transform."""
        if self.strategy == 'kf_scale':
            beta = kf_scale_dict.get(node_name + ".kf_scale").cpu()
            next_node = model.module_map[node_name].child_nodes[0]
            bn_weight = 1
            if next_node.module_type == "BatchNorm2d":
                bn_weight = next_node.module.weight.data.abs().cpu()
            score = bn_weight * (beta - (1 - beta)).squeeze()
            _, idx = score.sort()
            idx = idx.numpy().tolist()
            idx.reverse()
            select_idx = idx[:select_idx]
            return [1 if idx in select_idx else 0 for idx in range(node_channels)]
        elif self.strategy == 'l1':
            node = model.module_map[node_name]
            weight = node.module.weight.data.cpu()
            l1_norm = torch.norm(weight.view(len(weight), -1), p=1, dim=1)
            _, idx = l1_norm.sort()
            idx = idx.numpy().tolist()
            idx.reverse()
            select_idx = idx[:select_idx]
            return [1 if idx in select_idx else 0 for idx in range(node_channels)]
        else:
            return [1 if idx < select_idx else 0 for idx in range(node_channels)]


def compress(model, desc, strategy=None, prune_type='prune'):
    """Do compress model."""
    dag_cls = ClassFactory.get_cls(ClassType.NETWORK, "DAGFactory")
    dag_model = dag_cls(model=copy.deepcopy(model)).get_model()
    desc = decode(dag_model, desc, strategy)
    dag_model.insight_node_relations(desc)
    if prune_type == 'mask':
        return mask_model(copy.deepcopy(model), dag_model)
    return prune_model(copy.deepcopy(model), dag_model)
