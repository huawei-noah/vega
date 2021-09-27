# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""This is DAG Cell for network."""
import copy
import json
import random
import logging
from collections import OrderedDict
from vega.common import ClassFactory, ClassType, Config
from vega.core.search_space import SearchSpace
from vega.core.pipeline.conf import PipeStepConfig
from vega.model_zoo import ModelZoo
from vega.algorithms.nas.dag_mutate.search_blocks import search_blocks, forward_block_out_shape


@ClassFactory.register(ClassType.SEARCHSPACE)
class DAGMutateSearchSpace(SearchSpace):
    """Prune SearchSpace."""

    @classmethod
    def to_desc(self, desc):
        """Decode to model desc."""
        if not hasattr(self, "model") or not self.model:
            self.model = ModelZoo().get_model(PipeStepConfig.model.model_desc,
                                              PipeStepConfig.model.pretrained_model_file)
        model = copy.deepcopy(self.model)
        blocks = search_blocks(model)
        target_desc = OrderedDict(copy.deepcopy(model.to_desc()))
        return mutate_blocks(blocks, target_desc, desc)


def mutate_blocks(blocks, target_desc, block_desc):
    """Mutate Block."""
    target_block = generate_d_blocks(block_desc)
    logging.info("generate d block: {}".format(target_block))
    mutated_blocks = [block for block in blocks if
                      block.c_in == target_block.c_in and block.c_out == target_block.c_out]
    if not mutate_blocks:
        return None
    mutated_desc = target_desc
    for block in mutated_blocks:
        if random.uniform(0, 1) > 0.5:
            continue
        mutated_desc = mutate_block(mutated_desc, block, target_block)
    return mutated_desc


def generate_d_blocks(block_desc):
    """Generate d blocks."""
    block_str = block_desc.get("block_str")
    stride = block_desc.get('stride')
    c_in = block_desc.get('c_in')
    ops = block_desc.get('ops') or ['conv3', 'conv1', 'conv3_grp2', 'conv3_grp4', 'conv3_base1', 'conv3_base32',
                                    'conv3_sep']
    target_block = Config(dict(type="EncodedBlock", block_str=block_str, in_channel=c_in, op_names=ops, stride=stride))
    target_block.c_in = c_in
    target_block.c_out = forward_block_out_shape(target_block, [1, c_in, 32, 32], idx=1)
    return target_block


def mutate_block(model_desc, mutated_block, target_block=None):
    """Mutate block."""
    if not mutated_block.c_in or not mutated_block.c_out:
        return None
    if not (mutated_block.c_in == target_block.c_in and mutated_block.c_out == target_block.c_out):
        return None
    logging.info("Mutate blocks start module name: {}, end module name: {}".format(
        mutated_block.start_name, mutated_block.end_name))
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
