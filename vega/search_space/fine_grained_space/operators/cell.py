# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torch operators."""
import torch
import torch.nn as nn
from vega.core.common.class_factory import ClassType, ClassFactory
from vega.search_space.fine_grained_space.operators.mix_ops import OPS
from vega.search_space.fine_grained_space.operators.conv import conv_bn_relu


@ClassFactory.register(ClassType.SEARCH_SPACE)
class ContextualCell_v1(nn.Module):
    """New contextual cell design."""

    def __init__(self, op_names, config, inp, repeats=1, concat=False):
        """Construct ContextualCell_v1 class.

        :param op_names: list of operation indices
        :param config: list of config numbers
        :param inp: input channel
        :param repeats: number of repeated times
        :param concat: concat the result if set to True, otherwise add the result
        """
        super(ContextualCell_v1, self).__init__()
        self._ops = nn.ModuleList()
        self._pos = []
        self._collect_inds = [0]
        self._pools = ['x']
        for ind, op in enumerate(config):
            # first op is always applied on x
            if ind == 0:
                pos = 0
                op_id = op
                self._collect_inds.remove(pos)
                op_name = op_names[op_id]
                self._ops.append(OPS[op_name](inp, 1, True, repeats))  # turn-off scaling in batch norm
                self._pos.append(pos)
                self._collect_inds.append(ind + 1)
                self._pools.append('{}({})'.format(op_name, self._pools[pos]))
            else:
                pos1, pos2, op_id1, op_id2 = op
                # drop op_id from loose ends
                for ind2, (pos, op_id) in enumerate(zip([pos1, pos2], [op_id1, op_id2])):
                    if pos in self._collect_inds:
                        self._collect_inds.remove(pos)
                    op_name = op_names[op_id]
                    self._ops.append(OPS[op_name](inp, 1, True, repeats))  # turn-off scaling in batch norm
                    self._pos.append(pos)
                    # self._collect_inds.append(ind * 3 + ind2 - 1) # Do not collect intermediate
                    self._pools.append('{}({})'.format(op_name, self._pools[pos]))
                # summation
                op_name = 'sum'
                self._ops.append(AggregateCell(size_1=None, size_2=None, agg_size=inp, pre_transform=False,
                                               concat=concat))  # turn-off convbnrelu
                self._pos.append([ind * 3 - 1, ind * 3])
                self._collect_inds.append(ind * 3 + 1)
                self._pools.append('{}({},{})'.format(op_name, self._pools[ind * 3 - 1], self._pools[ind * 3]))

    def forward(self, x):
        """Do an inference on ContextualCell_v1.

        :param x: input tensor
        :return: output tensor
        """
        feats = [x]
        for pos, op in zip(self._pos, self._ops):
            if isinstance(pos, list):
                assert len(pos) == 2, "Two ops must be provided"
                feats.append(op(feats[pos[0]], feats[pos[1]]))
            else:
                feats.append(op(feats[pos]))
        out = 0
        for i in self._collect_inds:
            out += feats[i]
        return out


@ClassFactory.register(ClassType.SEARCH_SPACE)
class AggregateCell(nn.Module):
    """Aggregate two cells and sum or concat them up."""

    def __init__(self, size_1, size_2, agg_size, pre_transform=True, concat=False):
        """Construct AggregateCell.

        :param size_1: channel of first input
        :param size_2: channel of second input
        :param agg_size: channel of aggregated tensor
        :param pre_transform: whether to do a transform on two inputs
        :param concat: concat the result if set to True, otherwise add the result
        """
        super(AggregateCell, self).__init__()
        self.pre_transform = pre_transform
        self.concat = concat
        if self.pre_transform:
            self.branch_1 = conv_bn_relu(size_1, agg_size, 1, 1, 0)
            self.branch_2 = conv_bn_relu(size_2, agg_size, 1, 1, 0)
        if self.concat:
            self.conv1x1 = conv_bn_relu(agg_size * 2, agg_size, 1, 1, 0)

    def forward(self, x1, x2):
        """Do an inference on AggregateCell.

        :param x1: first input
        :param x2: second input
        :return: output
        """
        if self.pre_transform:
            x1 = self.branch_1(x1)
            x2 = self.branch_2(x2)
        if x1.size()[2:] > x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:], mode='bilinear', align_corners=True)(x2)
        elif x1.size()[2:] < x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:], mode='bilinear', align_corners=True)(x1)
        if self.concat:
            return self.conv1x1(torch.cat([x1, x2], 1))
        else:
            return x1 + x2
