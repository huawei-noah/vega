# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined parameter sharing cells."""
import tensorflow as tf
from vega.core.common import Config
from vega.search_space.networks import NetTypes, NetTypesMap, NetworkFactory
from vega.search_space.networks.tensorflow.blocks.operations import FactorizedReduce, ReluConvBn
from vega.search_space.networks.tensorflow.blocks.operations import Identity, drop_path


@NetworkFactory.register(NetTypes.BLOCK)
class MixedOp(object):
    """Mix operations between two nodes.

    :param desc: description of MixedOp
    :type desc: Config
    """

    def __init__(self, desc):
        """Init MixedOp."""
        self.C = desc.C
        self.stride = desc.stride
        self.ops_cands = desc.ops_cands
        self.data_format = desc.data_format

    def __call__(self, x, training, weights=None):
        """Forward function of MixedOp."""
        if not isinstance(self.ops_cands, list):
            op_desc = {'C': self.C, 'stride': self.stride, 'affine': True, 'data_format': self.data_format}
            class_op = NetworkFactory.get_network(
                NetTypesMap['block'], self.ops_cands)
            self._ops = class_op(Config(op_desc))
        else:
            self._ops = []
            for primitive in self.ops_cands:
                op_desc = {'C': self.C, 'stride': self.stride, 'affine': False, 'data_format': self.data_format}
                class_op = NetworkFactory.get_network(
                    NetTypesMap['block'], primitive)
                op = class_op(Config(op_desc))
                if 'pool' in primitive:
                    self._ops.append((op, True))
                else:
                    self._ops.append((op, False))
        if weights is not None:
            result = []
            for idx, (op, pool) in enumerate(self._ops):
                tmp = op(x, training=training)
                if pool:
                    tmp = tf.layers.batch_normalization(tmp, axis=1 if self.data_format == 'channels_first' else 3,
                                                        trainable=False, training=training)
                tmp = weights[idx] * tmp
                result.append(tmp)
            return tf.add_n(result)
        else:
            if isinstance(self._ops, list):
                for idx, (op, pool) in enumerate(self._ops):
                    x = op(x, training=training)
                    if pool:
                        x = tf.layers.batch_normalization(x, axis=1 if self.data_format == 'channels_first' else 3,
                                                          trainable=False, training=training)
            else:
                x = self._ops(x, training=training)
            return x


@NetworkFactory.register(NetTypes.BLOCK)
class Cell(object):
    """Cell structure according to desc.

    :param desc: description of Cell
    :type desc: Config
    """

    def __init__(self, desc):
        """Init Cell."""
        self.desc = desc

    def _pre_desc(self, channel_in, channel_out, kernel_size, stride, padding, affine, data_format):
        pre_desc = Config()
        pre_desc.channel_in = channel_in
        pre_desc.channel_out = channel_out
        pre_desc.affine = affine
        pre_desc.kernel_size = kernel_size
        pre_desc.stride = stride
        pre_desc.padding = padding
        pre_desc.data_format = data_format
        return pre_desc

    def _compile(self, C, op_names, indices_out, indices_inp, concat, reduction, data_format):
        """Compile the cell.

        :param C: channels of this cell
        :type C: int
        :param op_names: list of all the operations in description
        :type op_names: list of str
        :param indices_out: list of all output nodes
        :type indices_out: list of int
        :param indices_inp: list of all input nodes link to output node
        :type indices_inp: list of int
        :param concat: cell concat list of output node
        :type concat: list of int
        :param reduction: whether to reduce
        :type reduction: bool
        """
        self._concat = concat
        self._multiplier = len(concat)
        self._ops = []
        self.out_inp_list = []
        temp_list = []
        idx_cmp = 2
        for i in range(len(op_names)):
            if indices_out[i] == idx_cmp:
                temp_list.append(indices_inp[i])
            elif indices_out[i] > idx_cmp:
                self.out_inp_list.append(temp_list.copy())
                temp_list = []
                idx_cmp += 1
                temp_list.append(indices_inp[i])
            else:
                raise Exception("input index should not less than idx_cmp")
            stride = 2 if reduction and indices_inp[i] < 2 else 1
            op = self.build_mixedop(C=C, stride=stride, ops_cands=op_names[i], data_format=data_format)
            self._ops.append(op)
        self.out_inp_list.append(temp_list.copy())
        if len(self.out_inp_list) != self._steps:
            raise Exception("out_inp_list length should equal to steps")

    def build_mixedop(self, **kwargs):
        """Build MixedOp.

        :param kwargs: arguments for MixedOp
        :type kwargs: dict
        :return: MixedOp Object
        :rtype: MixedOp
        """
        mixedop_desc = Config(**kwargs)
        return MixedOp(mixedop_desc)

    def __call__(self, s0, s1, training, weights=None, drop_prob=0):
        """Forward function of Cell.

        :param s0: feature map of previous of previous cell
        :type s0: torch tensor
        :param s1: feature map of previous cell
        :type s1: torch tensor
        :param weights: weights of operations in cell
        :type weights: torch tensor, 2 dimension
        :return: cell output
        :rtype: torch tensor
        """
        desc = self.desc
        data_format = desc.data_format
        genotype = desc.genotype
        steps = desc.steps
        C_prev_prev = desc.C_prev_prev
        C_prev = desc.C_prev
        C = desc.C
        concat = desc.concat
        self.reduction = desc.reduction
        reduction_prev = desc.reduction_prev
        affine = True
        if isinstance(genotype[0][0], list):
            affine = False
        pre0_desc = self._pre_desc(C_prev_prev, C, 1, 1, 0, affine, data_format)
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(pre0_desc)
        else:
            self.preprocess0 = ReluConvBn(pre0_desc)
        pre1_desc = self._pre_desc(C_prev, C, 1, 1, 0, affine, data_format)
        self.preprocess1 = ReluConvBn(pre1_desc)
        self._steps = steps
        self.search = desc.search
        op_names, indices_out, indices_inp = zip(*genotype)
        self._compile(C, op_names, indices_out,
                      indices_inp, concat, self.reduction, data_format)
        s0 = self.preprocess0(s0, training=training)
        s1 = self.preprocess1(s1, training=training)
        drop = not self.search and drop_prob > 0.
        states = [s0, s1]
        idx = 0
        for i in range(self._steps):
            hlist = []
            for j, inp in enumerate(self.out_inp_list[i]):
                op = self._ops[idx + j]
                if weights is None:
                    h = op(states[inp], training=training)
                else:
                    h = op(states[inp], weights=weights[idx + j], training=training)
                if drop and not isinstance(op._ops.block, Identity):
                    h = drop_path(h, drop_prob)
                hlist.append(h)
            s = tf.add_n(hlist)
            states.append(s)
            idx += len(self.out_inp_list[i])
        axis = 1 if data_format == 'channels_first' else 3
        out = tf.concat([states[i] for i in self._concat], axis=axis)
        return out
