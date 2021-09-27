# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Prune DAG model."""
import vega

if vega.is_torch_backend():
    import torch


def prune_conv2d_out_channels(module, value):
    """Prune out channels of Conv2d."""
    assert sum(value) == module.out_channels
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
    assert sum(value) == module.in_channels
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


def prune_linear(module, value):
    """Prune linear."""
    if sum(value) == module.in_features:
        idx_in = [idx for idx, value in enumerate(value) if value == 1]
    else:
        idx_in = [idx for idx, value in enumerate([1] * module.in_features)]
    module.weight.data = module.weight.data[:, idx_in]


def prune_batch_norm(module, value):
    """Prune Batch Norm."""
    assert sum(value) == module.num_features
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
        if isinstance(node.module, torch.nn.Conv2d):
            if node.c_in:
                node.module.in_channels = sum(node.c_in)
                prune_conv2d_in_channels(node.module, node.c_in)
            if node.c_out:
                node.module.out_channels = sum(node.c_out)
                prune_conv2d_out_channels(node.module, node.c_out)
        elif isinstance(node.module, torch.nn.BatchNorm2d):
            if node.c_in:
                node.module.num_features = sum(node.c_in)
                node.c_out = node.c_in
                prune_batch_norm(node.module, node.c_in)
        elif isinstance(node.module, torch.nn.Linear):
            if node.c_in:
                if sum(node.c_in) == len(node.c_in):
                    continue
                if node.module.in_features == len(node.c_in):
                    node.module.in_features = sum(node.c_in)
                else:
                    node.module.in_features = node.module.in_features // len(node.c_in) * sum(node.c_in)
                prune_linear(node.module, node.c_in)
        elif node.module_type == 'torch_tensor_view':
            if node.c_in and len(node.c_in) != sum(node.c_in) and node.saved_args and len(node.saved_args) > 1:
                node.saved_args = tuple([node.saved_args[0], node.saved_args[1] // len(node.c_in) * sum(node.c_in)])
    return model
