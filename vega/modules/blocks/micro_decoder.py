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

"""This is SearchSpace for blocks."""
import sys
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module
from vega.modules.connections import ProcessList, Sequential
from vega.modules.operators import conv_bn_relu, conv3x3, conv_bn_relu6
from vega.modules.operators import AggregateCell, ContextualCell_v1
from vega.modules.operators import ops


@ClassFactory.register(ClassType.NETWORK)
class InvertedConv(Sequential):
    """Create InvertedConv SearchSpace."""

    def __init__(self, inp, oup, stride, kernel=3, expand_ratio=1):
        """Construct InvertedResidual class.

        :param inp: input channel
        :param oup: output channel
        :param stride: stride
        :param kernel: kernel
        :param expand_ratio: channel increase multiplier
        """
        hidden_dim = round(inp * expand_ratio)
        conv = []
        if expand_ratio > 1:
            conv = [
                conv_bn_relu6(C_in=inp, C_out=hidden_dim, kernel_size=1, stride=1, padding=0, inplace=True),
            ]
        conv = conv + [
            conv_bn_relu6(C_in=hidden_dim, C_out=hidden_dim, kernel_size=kernel, stride=stride, padding=kernel // 2,
                          groups=hidden_dim, depthwise=True, inplace=True),
            ops.Conv2d(in_channels=hidden_dim, out_channels=oup,
                       kernel_size=1, stride=1, padding=0, bias=False),
            ops.BatchNorm2d(num_features=oup)
        ]
        super(InvertedConv, self).__init__(*conv)


@ClassFactory.register(ClassType.NETWORK)
class InvertedResidual(Module):
    """Create InvertedResidual SearchSpace."""

    def __init__(self, inp, oup, stride, kernel=3, expand_ratio=1):
        """Construct InvertedResidual class.

        :param inp: input channel
        :param oup: output channel
        :param stride: stride
        :param kernel: kernel
        :param expand_ratio: channel increase multiplier
        """
        super(InvertedResidual, self).__init__()
        self.user_res_connect = stride == 1 and inp == oup
        self.conv = InvertedConv(inp, oup, stride, kernel, expand_ratio)

    def call(self, inputs):
        """Do an inference on InvertedResidual."""
        if self.user_res_connect:
            return inputs + self.conv(inputs)
        else:
            return self.conv(inputs)


class MergeCell(Module):
    """Pass two inputs through ContextualCell, and aggregate their results."""

    def __init__(self, op_names, ctx_config, conn, inps, agg_size, ctx_cell, repeats=1,
                 cell_concat=False):
        """Construct MergeCell class.

        :param op_names: list of operation indices
        :param ctx_config: list of config numbers
        :param conn: list of two indices
        :param inps: channel of first and second input
        :param agg_size: number of aggregation channel
        :param ctx_cell: ctx module
        :param repeats: number of repeats
        :param cell_concat: whether to concat or add cells
        """
        super(MergeCell, self).__init__()
        inp_1, inp_2 = inps
        self.op_1 = ctx_cell(
            op_names=op_names, config=ctx_config, inp=inp_1, repeats=repeats)
        self.op_2 = ctx_cell(
            op_names=op_names, config=ctx_config, inp=inp_2, repeats=repeats)
        self.agg = AggregateCell(
            size_1=inp_1, size_2=inp_2, agg_size=agg_size, concat=cell_concat)

    def call(self, x1, x2):
        """Do an inference on MergeCell.

        :param x1: input tensor 1
        :param x2: input tensor 2
        :return: output tensor
        """
        x1 = self.op_1(x1)
        x2 = self.op_2(x2)
        return self.agg(x1, x2)


@ClassFactory.register(ClassType.NETWORK)
class MicroDecoder_Upsample(Module):
    """Call torch.Upsample."""

    def __init__(self, collect_inds, agg_concat):
        super(MicroDecoder_Upsample, self).__init__()
        self.collect_inds = collect_inds
        self.agg_concat = agg_concat

    def call(self, x):
        """Forward x."""
        out = x[self.collect_inds[0]]
        for i in range(1, len(self.collect_inds)):
            collect = x[self.collect_inds[i]]
            if ops.get_shape(out)[2] > ops.get_shape(collect)[2]:
                # upsample collect
                collect = ops.interpolate(collect, size=ops.get_shape(
                    out)[2:], mode='bilinear', align_corners=True)
            elif ops.get_shape(collect)[2] > ops.get_shape(out)[2]:
                out = ops.interpolate(out, size=ops.get_shape(collect)[2:], mode='bilinear', align_corners=True)
            if self.agg_concat:
                out = ops.concat([out, collect])
            else:
                out += collect
        out = ops.Relu()(out)
        return out


@ClassFactory.register(ClassType.NETWORK)
class MicroDecoder(Module):
    """Parent class for MicroDecoders."""

    def __init__(self, backbone_out_sizes, op_names, num_classes, config, agg_size=64, aux_cell=False,
                 sep_repeats=1, agg_concat=False, num_pools=4, ctx_cell=ContextualCell_v1, cell_concat=False, **params):
        """Construct MicroDecoder class.

        :param op_names: list of operation candidate names
        :param backbone_out_sizes: backbone output channels
        :param num_classes: number of classes
        :param config: config list
        :param agg_size: number of channels in aggregation cells
        :param num_pools: number of pools
        :param ctx_cell: ctx module
        :param aux_cell: aux cells
        :param sep_repeats: number of repeats
        :param agg_concat: whether to concat or add agg results
        :param cell_concat: whether to concat or add cells
        :param params: other parameters
        """
        # NOTE: bring all outputs to the same size
        super(MicroDecoder, self).__init__()
        adapt = []
        for out_idx, size in enumerate(backbone_out_sizes):
            adapt.append(conv_bn_relu(C_in=size, C_out=agg_size,
                                      kernel_size=1, stride=1, padding=0, affine=True))
            backbone_out_sizes[out_idx] = agg_size
        if sys.version_info[0] < 3:
            backbone_out_sizes = list(backbone_out_sizes)
        else:
            backbone_out_sizes = backbone_out_sizes.copy()
        self.adapt = ProcessList(*adapt)
        cell_config, conns = config
        collect_inds = []
        cells = []

        for block_idx, conn in enumerate(conns):
            for ind in conn:
                if ind in collect_inds:
                    # remove from outputs if used by pool cell
                    collect_inds.remove(ind)
            ind_1, ind_2 = conn
            cells.append(MergeCell(op_names=op_names, ctx_config=cell_config, conn=conn,
                                   inps=(
                                       backbone_out_sizes[ind_1], backbone_out_sizes[ind_2]),
                                   agg_size=agg_size,
                                   ctx_cell=ctx_cell, repeats=sep_repeats,
                                   cell_concat=cell_concat))
            cell_concat = False
            collect_inds.append(block_idx + num_pools)
            backbone_out_sizes.append(agg_size)
            # for description

        self.block = ProcessList(*cells, out_list=conns)
        self.upsample = MicroDecoder_Upsample(
            collect_inds=collect_inds, agg_concat=agg_concat)
        self.pre_clf = conv_bn_relu(C_in=agg_size * (len(collect_inds) if agg_concat else 1),
                                    C_out=agg_size, kernel_size=1, stride=1, padding=0)
        self.conv_clf = conv3x3(
            inchannel=agg_size, outchannel=num_classes, stride=1, bias=True)


@ClassFactory.register(ClassType.NETWORK)
class Seghead(Module):
    """Class of seghead."""

    def __init__(self, shape):
        super(Seghead, self).__init__()
        self.head = ops.InterpolateScale(mode='bilinear', align_corners=True)
        self.shape = shape

    def call(self, inputs):
        """Forward x."""
        self.head.size = (self.shape, self.shape)
        return self.head(inputs)
