# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""This is Search all blocks in network."""
from collections import OrderedDict
from vega.common import ClassFactory, ClassType


def is_connection_node(node):
    """Determine is connection node."""
    return node.is_operator_conn_module or len(node.child_nodes) > 1 or node.module_type == 'torch_func_cat'


class BlockItems(object):
    """Blocks Items."""

    def __init__(self):
        self._nodes = OrderedDict()
        self._start_name = None
        self._end_name = None

    def add(self, name, node, start_node=False, end_node=False):
        """Add a node into items."""
        self._nodes[name] = node
        if start_node:
            self._start_name = name
        if end_node:
            self._end_name = name

    @property
    def nodes(self):
        """Get nodes."""
        return self._nodes

    @property
    def c_in(self):
        """Get input shape."""
        convs = [node for name, node in self.nodes.items() if node.module_type == 'Conv2d']
        if convs:
            return convs[-1].module.in_channels
        return None  # in_node

    @property
    def c_out(self):
        """Get output shape."""
        convs = [node for name, node in self.nodes.items() if node.module_type == 'Conv2d']
        if convs:
            return convs[0].module.out_channels
        return 256

    @property
    def start_name(self):
        """Get start name."""
        return self._start_name or next(iter(reversed(self._nodes)))

    @property
    def end_name(self):
        """Get end name."""
        return self._end_name or next(iter(self._nodes))


def search_blocks_items(in_node):
    """Search and list all sub blocks items."""
    items = BlockItems()
    c_nodes = [in_node]
    while c_nodes:
        node = c_nodes.pop()
        items.add(node.name, node)
        for parent_node in node.parent_nodes:
            if not is_connection_node(parent_node):
                c_nodes.append(parent_node)
            else:
                items.add(parent_node.name, parent_node, start_node=True)
    return items


def search_blocks(model):
    """Search all blocks of dag network."""
    blocks = []
    for name, node in model.named_nodes():
        if is_connection_node(node):
            blocks.append(search_blocks_items(node))
    return blocks


def forward_block_out_shape(block_desc, input_shape, idx=None):
    """Forward blocks."""
    import torch
    block = ClassFactory.get_instance(ClassType.NETWORK, block_desc)
    out_shape = block(torch.ones(*input_shape)).shape
    if idx:
        return out_shape[idx]
    return out_shape
