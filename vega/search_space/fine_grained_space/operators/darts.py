# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all darts operators."""
import torch
import torch.nn as nn
from vega.core.common.class_factory import ClassType, ClassFactory
from vega.search_space.fine_grained_space.operators.functional import FactorizedReduce, Identity, drop_path
from vega.search_space.fine_grained_space.operators.mix_ops import MixedOp
from vega.search_space.fine_grained_space.operators.conv import ReLUConvBN


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Cell(nn.Module):
    """Cell structure according to desc."""

    reduction = None
    reduction_prev = None
    C_prev_prev = None
    C_prev = None
    C = None
    concat_size = 0

    def __init__(self, desc):
        """Init Cell."""
        super(Cell, self).__init__()
        self.desc = desc
        self.drop_path_prob = desc.get('drop_path_prob') or 0

    def build(self):
        """Build Cell."""
        affine = True
        if isinstance(self.desc.genotype[0][0], list):
            affine = False
        if self.reduction_prev:
            self.preprocess0 = FactorizedReduce(self.C_prev_prev, self.C, affine)
        else:
            self.preprocess0 = ReLUConvBN(self.C_prev_prev, self.C, 1, 1, 0, affine)
        self.preprocess1 = ReLUConvBN(self.C_prev, self.C, 1, 1, 0, affine)
        self._steps = self.desc.steps
        self.search = self.desc.search
        self._ops = nn.ModuleList()
        op_names, indices_out, indices_inp = zip(*self.desc.genotype)
        self._compile(self.C, op_names, indices_out, indices_inp, self.desc.concat, self.reduction)
        self.concat_size = len(self.desc.concat)
        return self

    def _compile(self, C, op_names, indices_out, indices_inp, concat, reduction):
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
            op = MixedOp(C=C, stride=stride, ops_cands=op_names[i])
            self._ops.append(op)
        self.out_inp_list.append(temp_list.copy())
        if len(self.out_inp_list) != self._steps:
            raise Exception("out_inp_list length should equal to steps")

    def forward(self, s0, s1, weights=None, drop_prob=0):
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
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        drop = not self.search and drop_prob > 0.
        states = [s0, s1]
        idx = 0
        for i in range(self._steps):
            hlist = []
            for j, inp in enumerate(self.out_inp_list[i]):
                op = self._ops[idx + j]
                if weights is None:
                    h = op(states[inp])
                else:
                    h = op(states[inp], weights[idx + j])
                if drop and not isinstance(op._ops.block, Identity):
                    h = drop_path(h, drop_prob)
                hlist.append(h)
            s = sum(hlist)
            states.append(s)
            idx += len(self.out_inp_list[i])
        return torch.cat([states[i] for i in self._concat], dim=1)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class PreOneStem(nn.Module):
    """Class of one stem convolution.

    :param desc: description of PreOneStem
    :type desc: Config
    """

    def __init__(self, desc):
        """Init PreOneStem."""
        super(PreOneStem, self).__init__()
        self._C = desc.C
        self._stem_multi = desc.stem_multi
        self.C_curr = self._stem_multi * self._C
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.C_curr)
        )

    def forward(self, x):
        """Forward function of PreOneStem."""
        x = self.stem(x)
        return x, x


class AuxiliaryHead(nn.Module):
    """Auxiliary Head of Network.

    :param C: input channels
    :type C: int
    :param num_classes: numbers of classes
    :type num_classes: int
    :param input_size: input size
    :type input_size: int
    """

    def __init__(self, C, num_classes, input_size):
        """Init AuxiliaryHead."""
        super(AuxiliaryHead, self).__init__()
        s = input_size - 5
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=s, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        """Forward function of Auxiliary Head."""
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
