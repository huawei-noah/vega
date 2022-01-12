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
"""This is DAG Cell for network."""
import random
import logging
import numpy as np
from vega.common.config import Config
from vega.common.class_factory import ClassFactory, ClassType
from vega.metrics.flops_and_params import calc_model_flops_params
from vega.algorithms.nas.dnet_nas.dblock_nas_codec import decode_d_block_str


def forward_block_out_shape(block_desc, input_shape, idx=None, cal_flops_and_params=False):
    """Forward blocks."""
    import torch
    block = ClassFactory.get_instance(ClassType.NETWORK, block_desc)
    x = torch.ones(*input_shape)
    out_shape = block(x).shape
    if idx:
        out_shape = out_shape[idx]
    if cal_flops_and_params:
        flops, params = calc_model_flops_params(block, x)
        return out_shape, flops * 1e-9, params * 1e-6
    return out_shape


class BlockGenerator(object):
    """Block generator."""

    def __init__(self, c_in, block_type='GhostModule'):
        self.c_in = c_in
        self.c_out = None
        self.flops = None
        self.params = None
        self.block_type = block_type
        self.each_block_max_samples = 10

    def _gen_one_block(self):
        if self.block_type == 'DBlock':
            return gen_d_blocks(self.c_in)
        elif self.block_type == 'DAGGraphCell':
            return gen_graph_cell(self.c_in)
        return gen_ghost_block(self.c_in)

    def run(self, block):
        """Run generator."""
        flops, params = block.flops_params
        for _ in range(self.each_block_max_samples):
            target_block = self._gen_one_block()
            if target_block and block.c_in == target_block.c_in and block.c_out == target_block.c_out:
                if self._params_filter(target_block, params):
                    return target_block

    @classmethod
    def _flops_filter(cls, target_block, flops):
        return 0.5 * flops < target_block.c_flops < 2 * flops

    @classmethod
    def _params_filter(cls, target_block, params):
        return 0.5 * params < target_block.c_params < 1.5 * params


def gen_ghost_block(c_in):
    """Generate Ghost block."""
    block_str = random.choice([1, 2, 3])
    planes = random.choice([16, 32, 64, 128, 256, 512])
    stride = random.choice([1, 2])
    target_block = Config(dict(type="GhostModule", inplanes=c_in, planes=planes, blocks=block_str, stride=stride))
    target_block.c_in = c_in
    target_block.c_out, target_block.c_flops, target_block.c_params = forward_block_out_shape(
        target_block, [2, c_in, 32, 32], 1, True)
    return target_block


def gen_d_blocks(c_in):
    """Generate d blocks."""
    try:
        op_choices = 7
        channel_choices = 5
        op_num = random.choice([1, 2, 3])
        skip_num = random.choice([0, 1])
        block_str = decode_d_block_str(op_choices, channel_choices, op_num, skip_num)
        stride = random.choice([1, 2])
        ops = ['conv3', 'conv1', 'conv3_grp2', 'conv3_grp4', 'conv3_base1', 'conv3_base32', 'conv3_sep']
        target_block = Config(
            dict(type="EncodedBlock", block_str=block_str, in_channel=c_in, op_names=ops, stride=stride))
        target_block.c_in = c_in
        target_block.c_out, target_block.c_flops, target_block.c_params = forward_block_out_shape(
            target_block, [2, c_in, 32, 32], 1, True)
        return target_block
    except Exception as ex:
        logging.debug("Failed to generate D blocks. ex={}".format(ex))
        return None


def gen_graph_cell(c_in):
    """Generate graph cell."""
    try:
        matrix_deep = random.choice([3, 4, 5, 6, 7])
        adj_matrix = np.array(np.random.randint(2, size=(matrix_deep, matrix_deep)))
        nodes = ['Input', 'Conv1x1BnRelu', 'Conv3x3BnRelu', 'Conv3x3BnRelu', 'Conv3x3BnRelu', 'MaxPool3x3',
                 'Output']
        out_channels = random.choice([64, 128, 256, 512, 1024])
        target_block = Config(
            dict(type='DagGraphCell', adj_matrix=adj_matrix, nodes=nodes, in_channels=c_in,
                 out_channels=out_channels))
        target_block.c_in = c_in
        target_block.c_out = forward_block_out_shape(target_block, [1, c_in, 32, 32], idx=1)
        return target_block
    except Exception as ex:
        logging.debug("Failed to generate Graph Cell. ex={}".format(ex))
        return None
