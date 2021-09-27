# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Dag relations."""
import vega

if vega.is_torch_backend():
    import torch
    from torch.nn import Conv2d, Linear
elif vega.is_ms_backend():
    from mindspore.nn import Conv2d, Dense as Linear


def is_conv2d(module):
    """Determine Conv2d."""
    # depth-wise convolution not in pruned search space.
    return isinstance(module, Conv2d) and not is_depth_wise_conv(module)


def is_depth_wise_conv(module):
    """Determine Conv2d."""
    if hasattr(module, "groups"):
        return module.groups != 1 and module.in_channels == module.out_channels
    elif hasattr(module, "group"):
        return module.group != 1 and module.in_channels == module.out_channels


def is_connection_node(node):
    """Determine is connection node."""
    return node.is_operator_conn_module or len(node.child_nodes) > 1 or node.module_type == 'torch_func_cat'


def reset_c_out_node(node):
    """Determine is connection node."""
    if isinstance(node.module, Linear):
        return None
    else:
        return node.c_out


def check_and_export_model(pruned_model, dummy_input):
    """Check and export model to onnx file."""
    dummy_input = dummy_input or torch.ones(1, 3, 224, 224)
    torch.onnx.export(pruned_model, dummy_input, "pruned.onnx")


def sub_blocks_relation_search(in_node, c_out=None):
    """Search relations of blocks."""
    nodes_in_block = []
    c_nodes = [in_node]
    while c_nodes:
        node = c_nodes.pop()
        nodes_in_block.append(node)
        if isinstance(node.module, Conv2d):
            continue
        for parent_node in node.parent_nodes:
            if is_connection_node(parent_node):
                c_out = parent_node.c_out
            else:
                c_nodes.append(parent_node)
    for node in nodes_in_block:
        if not isinstance(node.module, Conv2d):
            node.c_in = c_out
        node.c_out = c_out
    return nodes_in_block


def sub_cat_relation_search(in_node, c_out=None):
    """Search relations of blocks."""
    nodes_in_block = []
    c_nodes = [in_node]
    while c_nodes:
        node = c_nodes.pop()
        nodes_in_block.append(node)
        for parent_node in node.parent_nodes:
            if not is_connection_node(parent_node):
                c_nodes.append(parent_node)
    if vega.is_torch_backend():
        for node in nodes_in_block:
            if is_conv2d(node.module):
                break
            if not isinstance(node.module, Conv2d):
                node.c_in = c_out
            node.c_out = c_out
    elif vega.is_ms_backend():
        for node in nodes_in_block[1:]:
            if node.child_nodes:
                node.c_out = node.child_nodes[0].c_in
            if not isinstance(node.module, Conv2d):
                node.c_in = node.c_out
    return nodes_in_block


def node_relations_search(model, desc):
    """Search relations of dag node."""
    for name, node in model.named_nodes():
        c_out = desc.get(node.name + '.out_channels')
        if c_out and not node.c_out:
            node.c_out = c_out
        else:
            for parent_node in node.parent_nodes:
                if node.module_type == 'torch_func_cat':
                    cat_c_outs = []
                    for n in node.parent_nodes:
                        cat_c_outs.extend(n.c_out)
                    node.c_out = cat_c_outs
                else:
                    node.c_out = min(parent_node.c_out, node.c_out) if node.c_out else parent_node.c_out
                if is_connection_node(parent_node):
                    break
        node.c_out = reset_c_out_node(node)
        for child_node in node.child_nodes:
            if not child_node.c_in:
                child_node.c_in = node.c_out
        if is_connection_node(node):
            if node.module_type == 'torch_func_cat':
                sub_cat_relation_search(node, node.c_out)
            else:
                sub_blocks_relation_search(node, node.c_out)
    return model
