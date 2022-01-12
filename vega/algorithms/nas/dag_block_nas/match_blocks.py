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
"""This is Match all blocks in network."""
import json
import copy
import logging
from collections import OrderedDict
from vega.core.search_space.search_space import SearchSpace
from vega.algorithms.nas.dag_block_nas.block_generator import BlockGenerator
from vega.metrics.forward_latency import calc_forward_latency
from vega.common.class_factory import ClassFactory, ClassType


def is_connection_node(node):
    """Determine is connection node."""
    return node.is_operator_conn_module or len(node.child_nodes) > 1 or node.module_type == 'torch_func_cat'


def check_latency(target_model, dummy_inputs=None):
    """Validate network."""
    try:
        import torch
        x = dummy_inputs or torch.ones(1, 3, 224, 224)
        calc_forward_latency(target_model, x)
    except Exception as ex:
        logging.info("sampled network is invalidate, ex={}".format(str(ex)))
        return None
    return True


class BlockItems(object):
    """Blocks Items."""

    def __init__(self):
        self._nodes = OrderedDict()
        self._start_name = None
        self._end_name = None

    def add(self, name, node, start_node=False, end_node=False):
        """Add a node into items."""
        self._nodes[name] = node
        self._nodes.move_to_end(name, last=False)
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
        res = self._filter_modules('Conv2d')
        if not res:
            return None
        return res[0].in_channels

    def _filter_modules(self, module_type):
        res = []
        for name, node in self.nodes.items():
            if not hasattr(node.module, 'named_modules'):
                continue
            for module_name, module in node.module.named_modules():
                if module.__class__.__name__ == module_type:
                    res.append(module)
        return res

    @property
    def c_out(self):
        """Get output shape."""
        res = self._filter_modules('Conv2d')
        if not res:
            return None
        return res[-1].out_channels

    @property
    def start_name(self):
        """Get start name."""
        return self._start_name or next(iter(self.nodes))

    @property
    def end_name(self):
        """Get end name."""
        return self._end_name or next(iter(reversed(self._nodes)))

    @property
    def flops_params(self):
        """Get Flops and Params."""
        dag_cls = ClassFactory.get_cls(ClassType.NETWORK, 'DagNetworkTorch')
        block = dag_cls(self._nodes)
        from vega.metrics.flops_and_params import calc_model_flops_params
        import torch
        x = torch.ones(2, self.c_in, 32, 32)
        flops, params = calc_model_flops_params(block, x)
        return flops * 1e-9, params * 1e-6


def match_blocks_items(in_node):
    """Match and list all sub blocks items."""
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


def match_blocks(model):
    """Match all blocks of dag network."""
    blocks = []
    for name, node in model.named_nodes():
        if is_connection_node(node):
            blocks.append(match_blocks_items(node))
    return blocks


def mutate_sub_blocks(block, target_desc, block_type_iter):
    """Mutate Sub Blocks."""
    if not block.c_in or not block.c_out:
        return target_desc
    block_type = next(block_type_iter)  # random.choice(block_type)
    target_block = BlockGenerator(c_in=block.c_in, block_type=block_type).run(block)

    if target_block:
        return mutate_block(target_desc, block, target_block)
    return target_desc


def mutate_block(model_desc, mutated_block, target_block=None):
    """Mutate block."""
    logging.info("Mutate blocks start module name: {}, end module name: {}, target module desc: {}:".format(
        mutated_block.start_name, mutated_block.end_name, target_block))
    mutated_map = OrderedDict()
    while model_desc:
        name, node = model_desc.popitem(0)
        if name != 'type':
            node = json.loads(node) if isinstance(node, str) else node
        if name not in mutated_block.nodes:
            mutated_map[name] = node
            continue
        if name == mutated_block.end_name:
            mutated_map[name] = dict(name=name, module=target_block, module_type=target_block.get("type"),
                                     parent_node_names=[mutated_block.start_name],
                                     child_node_names=node.get("child_node_names"))
        elif name == mutated_block.start_name:
            tmp_node = copy.deepcopy(node)
            tmp_node["child_node_names"] = [mutated_block.end_name]
            tmp_node["child_nodes"] = []
            mutated_map[name] = tmp_node
    return mutated_map


class SpaceIterRecord(object):
    """Record Space iter."""

    __records__ = []

    @classmethod
    def add_record(cls, record):
        """Add one record."""
        cls.__records__.append(record)

    @classmethod
    def clear(cls):
        """Clear records."""
        cls.__records__ = []

    @classmethod
    def dump(cls, file_path):
        """Dump record."""
        with open("{}/dag_block_nas.json".format(file_path), 'w') as f:
            json.dump(cls.__records__, f)
        cls.clear()


class SpaceIter(object):
    """Get Space iter."""

    def __init__(self, search_space, name):
        self.name = name
        self.space = list(filter(lambda x: x.get("key") == name, search_space.get("hyperparameters")))
        self.search_space = SearchSpace(dict(hyperparameters=self.space))

    def __iter__(self):
        """Get iter."""
        return self

    def __next__(self):
        """Get next sample."""
        res = self.search_space.sample()
        SpaceIterRecord.add_record(res)
        return res.get(self.name)
