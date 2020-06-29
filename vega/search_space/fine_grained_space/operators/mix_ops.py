# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torch operators."""
import torch.nn as nn
from vega.search_space.fine_grained_space.operators.conv import SepConv, DilConv, GAPConv1x1, conv1X1, conv3x3, \
    conv5x5, conv7x7
from vega.search_space.fine_grained_space.operators.functional import Zero, FactorizedReduce, Input
from vega.core.common.class_factory import ClassType, ClassFactory

OPS = {
    'none': lambda C, stride, affine, repeats=1: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine, repeats=1: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine, repeats=1: nn.MaxPool2d(
        3, stride=stride, padding=1),
    'global_average_pool': lambda C, stride, affine, repeats=1: GAPConv1x1(C, C),
    'skip_connect': lambda C, stride, affine, repeats=1: Input() if stride == 1 else FactorizedReduce(
        C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine, repeats=1: SepConv(C, C, 3, stride, 1, affine=affine, repeats=repeats),
    'sep_conv_5x5': lambda C, stride, affine, repeats=1: SepConv(C, C, 5, stride, 2, affine=affine, repeats=repeats),
    'sep_conv_7x7': lambda C, stride, affine, repeats=1: SepConv(C, C, 7, stride, 3, affine=affine, repeats=repeats),
    'dil_conv_3x3': lambda C, stride, affine, repeats=1: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine, repeats=1: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine, repeats=1: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)),
    'conv1x1': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv1X1(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv5x5': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv5x5(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv7x7': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv7x7(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3_dil2': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride, dilation=2),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3_dil3': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride, dilation=3),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3_dil12': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride, dilation=12),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'sep_conv_3x3_dil3': lambda C, stride, affine, repeats=1: SepConv(
        C, C, 3, stride, 3, affine=affine, dilation=3, repeats=repeats),
    'sep_conv_5x5_dil6': lambda C, stride, affine, repeats=1: SepConv(
        C, C, 5, stride, 12, affine=affine, dilation=6, repeats=repeats)
}


@ClassFactory.register(ClassType.SEARCH_SPACE)
class MixedOp(nn.Module):
    """Mix operations between two nodes.

    :param desc: description of MixedOp
    :type desc: Config
    """

    def __init__(self, C, stride, ops_cands):
        """Init MixedOp."""
        super(MixedOp, self).__init__()
        if not isinstance(ops_cands, list):
            self._ops = OPS[ops_cands](C, stride, True)
        else:
            self._ops = nn.ModuleList()
            for primitive in ops_cands:
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                self._ops.append(op)

    def forward(self, x, weights=None):
        """Forward function of MixedOp."""
        if weights is not None:
            return sum(w * op(x) for w, op in zip(weights, self._ops) if w != 0)
        else:
            return self._ops(x)
