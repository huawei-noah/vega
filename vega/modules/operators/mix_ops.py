# -*- coding: utf-8 -*-

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

"""Import all torch operators."""
from vega.common import ClassType, ClassFactory
from vega.modules.operators import Seq, SeparatedConv, DilConv, GAPConv1x1, conv1X1, \
    conv3x3, conv5x5, conv7x7, FactorizedReduce
from vega.modules.operators import ops

OPS = {
    'none': lambda C, stride, affine, repeats=1: ops.Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine, repeats=1: ops.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine, repeats=1: ops.MaxPool2d(
        3, stride=stride, padding=1),
    'global_average_pool': lambda C, stride, affine, repeats=1: Seq(GAPConv1x1(C, C)),
    'skip_connect': lambda C, stride, affine, repeats=1: ops.Identity() if stride == 1 else FactorizedReduce(
        C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine, repeats=1: SeparatedConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine, repeats=1: SeparatedConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine, repeats=1: SeparatedConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine, repeats=1: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine, repeats=1: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine, repeats=1: Seq(
        ops.Relu(inplace=False),
        ops.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        ops.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        ops.BatchNorm2d(C, affine=affine)),
    'conv1x1': lambda C, stride, affine, repeats=1: Seq(
        conv1X1(C, C, stride=stride),
        ops.BatchNorm2d(C, affine=affine),
        ops.Relu(inplace=False)),
    'conv3x3': lambda C, stride, affine, repeats=1: Seq(
        conv3x3(C, C, stride=stride),
        ops.BatchNorm2d(C, affine=affine),
        ops.Relu(inplace=False)),
    'conv5x5': lambda C, stride, affine, repeats=1: Seq(
        conv5x5(C, C, stride=stride),
        ops.BatchNorm2d(C, affine=affine),
        ops.Relu(inplace=False)),
    'conv7x7': lambda C, stride, affine, repeats=1: Seq(
        conv7x7(C, C, stride=stride),
        ops.BatchNorm2d(C, affine=affine),
        ops.Relu(inplace=False)),
    'conv3x3_dil2': lambda C, stride, affine, repeats=1: Seq(
        conv3x3(C, C, stride=stride, dilation=2),
        ops.BatchNorm2d(C, affine=affine),
        ops.Relu(inplace=False)),
    'conv3x3_dil3': lambda C, stride, affine, repeats=1: Seq(
        conv3x3(C, C, stride=stride, dilation=3),
        ops.BatchNorm2d(C, affine=affine),
        ops.Relu(inplace=False)),
    'conv3x3_dil12': lambda C, stride, affine, repeats=1: Seq(
        conv3x3(C, C, stride=stride, dilation=12),
        ops.BatchNorm2d(C, affine=affine),
        ops.Relu(inplace=False)),
    'sep_conv_3x3_dil3': lambda C, stride, affine, repeats=1: SeparatedConv(
        C, C, 3, stride, 3, affine=affine, dilation=3),
    'sep_conv_5x5_dil6': lambda C, stride, affine, repeats=1: SeparatedConv(
        C, C, 5, stride, 12, affine=affine, dilation=6)
}


@ClassFactory.register(ClassType.NETWORK)
class MixedOp(ops.Module):
    """Mix operations between two nodes.

    :param desc: description of MixedOp
    :type desc: Config
    """

    def __init__(self, C, stride, ops_cands):
        """Init MixedOp."""
        super(MixedOp, self).__init__()
        if not isinstance(ops_cands, list):
            # train
            self.add_module(ops_cands, OPS[ops_cands](C, stride, True))
        else:
            # search
            for primitive in ops_cands:
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = Seq(op, ops.BatchNorm2d(C, affine=False))
                self.add_module(primitive, op)

    def call(self, x, weights=None, selected_idx=None, *args, **kwargs):
        """Call function of MixedOp."""
        if selected_idx is None:
            if weights is None:
                for model in self.children():
                    x = model(x)
                return x
            weight_sum = ()
            for idx, op in enumerate(self.children()):
                weight_sum += (weights[idx] * op(x),)

            return ops.add_n(weight_sum)
        else:
            # SGAS alg: unchosen operations are pruned
            return list(self.children())[selected_idx](x)
