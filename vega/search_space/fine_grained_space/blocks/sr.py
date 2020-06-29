# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for blocks."""
import sys
import math
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.fine_grained_space.fine_grained_space import FineGrainedSpace
from vega.search_space.fine_grained_space.conditions import Sequential, Concat, Add, Process_list
from vega.search_space.fine_grained_space.operators import op, conv_bn_relu, AggregateCell, ChannelShuffle, \
    ContextualCell_v1, MicroDecoder_Upsample, conv3x3


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Block_Base(FineGrainedSpace):
    """Create Block_Base SearchSpace."""

    def constructor(self, inChannels, growRate, sh_groups, conv_groups, kSize=3):
        """Create Block_Base Block.

        :param inChannels: channel number of input
        :type inChannels: int
        :param growRate: growth rate of block
        :type growRate: int
        :param sh_groups: group number of shuffle operation
        :type sh_groups: int
        :param conv_groups: group number of convolution operation
        :type conv_groups: int
        :param kSize: kernel size of convolution operation
        :type kSize: int
        """
        self.channel_shuffle = ChannelShuffle(sh_groups)
        self.conv = op.Conv2d(in_channels=inChannels, out_channels=growRate, kernel_size=kSize,
                              stride=1, padding=(kSize - 1) // 2, groups=conv_groups)
        self.relu = op.ReLU()


@ClassFactory.register(ClassType.SEARCH_SPACE)
class InConv_Group(FineGrainedSpace):
    """Create InConv_Group SearchSpace."""

    def constructor(self, InChannel, OutChannel, growRate, nConvLayers, kSize=3):
        """Create InConv_Group Block.

        :param InChannel: channel number of input
        :type InChannel: int
        :param OutChannel: channel number of output
        :type OutChannel: int
        :param growRate: growth rate of block
        :type growRate: int
        :param nConvLayers: the number of convlution layer
        :type nConvLayers: int
        :param kSize: kernel size of convolution operation
        :type kSize: int
        """
        if InChannel != growRate:
            self.InConv = op.Conv2d(in_channels=InChannel, out_channels=growRate, kernel_size=1, stride=1, padding=0)
            self.make_conv(nConvLayers, growRate, OutChannel)
        else:
            self.make_conv(nConvLayers, growRate, OutChannel)

    def make_conv(self, nConvLayers, growRate, OutChannel):
        """Make Block_Base.

        :param OutChannel: channel number of output
        :type OutChannel: int
        :param growRate: growth rate of block
        :type growRate: int
        :param nConvLayers: the number of convlution layer
        :type nConvLayers: int
        """
        convs = []
        for c in range(nConvLayers):
            conv = Block_Base(inChannels=(c + 1) * growRate, growRate=growRate,
                              sh_groups=c + 1, conv_groups=min(4, 2 ** int(math.log(c + 1, 2))))
            cat = Concat(conv)
            convs.append(cat)
        self.seq = Sequential(*tuple(convs))
        self.LFF = op.Conv2d(in_channels=(nConvLayers + 1) * growRate, out_channels=OutChannel,
                             kernel_size=1, stride=1, padding=0)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Group_RDB(FineGrainedSpace):
    """Create Group_RDB SearchSpace."""

    def constructor(self, InChannel, OutChannel, growRate, nConvLayers, kSize=3):
        """Create Group_RDB Block.

        :param InChannel: channel number of input
        :type InChannel: int
        :param OutChannel: channel number of output
        :type OutChannel: int
        :param growRate: growth rate of block
        :type growRate: int
        :param nConvLayers: the number of convlution layer
        :type nConvLayers: int
        :param kSize: kernel size of convolution operation
        :type kSize: int
        """
        inter = InConv_Group(InChannel=InChannel, OutChannel=OutChannel, growRate=growRate,
                             nConvLayers=nConvLayers, kSize=kSize)
        if OutChannel == InChannel:
            self.merge = Add(inter)
        elif OutChannel == growRate:
            conv = op.Conv2d(in_channels=InChannel, out_channels=growRate, kernel_size=1, stride=1, padding=0)
            self.merge = Add(conv, inter)
        else:
            outconv = op.Conv2d(in_channels=InChannel, out_channels=OutChannel, kernel_size=1, stride=1, padding=0)
            self.merge = Add(outconv, inter)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Shrink_RDB(FineGrainedSpace):
    """Create Shrink_RDB SearchSpace."""

    def constructor(self, InChannel, OutChannel, growRate, nConvLayers, kSize=3):
        """Create Shrink_RDB Block.

        :param InChannel: channel number of input
        :type InChannel: int
        :param OutChannel: channel number of output
        :type OutChannel: int
        :param growRate: growth rate of block
        :type growRate: int
        :param nConvLayers: the number of convlution layer
        :type nConvLayers: int
        :param kSize: kernel size of convolution operation
        :type kSize: int
        """
        shrink_conv = op.Shrink_Conv(InChannel=InChannel, OutChannel=OutChannel,
                                     growRate=growRate, nConvLayers=nConvLayers, kSize=kSize)
        if OutChannel == InChannel:
            self.merge = Add(shrink_conv)
        elif OutChannel == growRate:
            InConv = op.Conv2d(in_channels=InChannel, out_channels=growRate, kernel_size=1, stride=1, padding=0)
            self.merge = Add(InConv, shrink_conv)
        else:
            OutConv = op.Conv2d(in_channels=InChannel, out_channels=OutChannel, kernel_size=1, stride=1, padding=0)
            self.merge = Add(OutConv, shrink_conv)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Cont_Seq(FineGrainedSpace):
    """Create Cont_Seq SearchSpace."""

    def constructor(self, InChannel, OutChannel, growRate, nConvLayers, kSize=3):
        """Create Cont_Seq Block.

        :param InChannel: channel number of input
        :type InChannel: int
        :param OutChannel: channel number of output
        :type OutChannel: int
        :param growRate: growth rate of block
        :type growRate: int
        :param nConvLayers: the number of convlution layer
        :type nConvLayers: int
        :param kSize: kernel size of convolution operation
        :type kSize: int
        """
        if InChannel != growRate:
            self.InConv = op.Conv2d(in_channels=InChannel, out_channels=growRate, kernel_size=1, stride=1, padding=0)
            self.pool = op.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool = op.AvgPool2d(kernel_size=2, stride=2)
        self.cont_conv = op.Cont_Conv(InChannel=InChannel, OutChannel=OutChannel,
                                      growRate=growRate, nConvLayers=nConvLayers, kSize=kSize)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Cont_RDB(FineGrainedSpace):
    """Create Cont_RDB SearchSpace."""

    def constructor(self, InChannel, OutChannel, growRate, nConvLayers, kSize=3):
        """Create Cont_RDB Block.

        :param InChannel: channel number of input
        :type InChannel: int
        :param OutChannel: channel number of output
        :type OutChannel: int
        :param growRate: growth rate of block
        :type growRate: int
        :param nConvLayers: the number of convlution layer
        :type nConvLayers: int
        :param kSize: kernel size of convolution operation
        :type kSize: int
        """
        cont_seq = Cont_Seq(InChannel=InChannel, OutChannel=OutChannel,
                            growRate=growRate, nConvLayers=nConvLayers, kSize=kSize)
        if OutChannel == InChannel:
            self.merge = Add(cont_seq)
        elif OutChannel == growRate:
            InConv = op.Conv2d(in_channels=InChannel, out_channels=growRate, kernel_size=1, stride=1, padding=0)
            self.merge = Add(InConv, cont_seq)
        else:
            OutConv = op.Conv2d(in_channels=InChannel, out_channels=OutChannel, kernel_size=1, stride=1, padding=0)
            self.merge = Add(OutConv, cont_seq)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class UPNet(FineGrainedSpace):
    """Create UPNet SearchSpace."""

    def constructor(self, scale, G0, kSize, n_colors):
        """Create UPNet Block.

        :param scale: scale
        :type scale: int
        :param G0: G0
        :type G0: G0
        :param kSize: kSize
        :type kSize: int
        :param n_colors: n_colors
        :type n_colors: int
        """
        if scale == 2 or scale == 3:
            self.conv = op.Conv2d(in_channels=G0, out_channels=G0 * 3, kernel_size=kSize,
                                  stride=1, padding=(kSize - 1) // 2)
            self.pixelshuffle = op.PixelShuffle(upscale_factor=scale)
            self.conv2 = op.Conv2d(in_channels=int(G0 * 3 / 4), out_channels=n_colors, kernel_size=kSize,
                                   stride=1, padding=(kSize - 1) // 2)
        elif scale == 4:
            self.conv = op.Conv2d(in_channels=G0, out_channels=G0 * 4, kernel_size=kSize,
                                  stride=1, padding=(kSize - 1) // 2)
            self.pixelshuffle = op.PixelShuffle(upscale_factor=2)
            self.conv2 = op.Conv2d(in_channels=G0, out_channels=G0 * 4, kernel_size=kSize,
                                   stride=1, padding=(kSize - 1) // 2)
            self.pixelshuffle2 = op.PixelShuffle(upscale_factor=2)
            self.conv3 = op.Conv2d(in_channels=G0, out_channels=n_colors, kernel_size=kSize,
                                   stride=1, padding=(kSize - 1) // 2)
        else:
            raise ValueError("scale must be 2 or 3 or 4.")


@ClassFactory.register(ClassType.SEARCH_SPACE)
class InvertedConv(FineGrainedSpace):
    """Create InvertedConv SearchSpace."""

    def constructor(self, inp, oup, stride, kernel=3, expand_ratio=1):
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
                op.Conv2d(in_channels=inp, out_channels=hidden_dim,
                          kernel_size=1, stride=1, padding=0, bias=False),
                op.BatchNorm2d(num_features=hidden_dim),
                op.ReLU6(inplace=True)
            ]
        conv = conv + [
            op.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel,
                      stride=stride, padding=kernel // 2, groups=hidden_dim, bias=False),
            op.BatchNorm2d(num_features=hidden_dim),
            op.ReLU6(inplace=True),
            op.Conv2d(in_channels=hidden_dim, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
            op.BatchNorm2d(num_features=oup)
        ]
        self.Conv = Sequential(*tuple(conv))


@ClassFactory.register(ClassType.SEARCH_SPACE)
class InvertedResidual(FineGrainedSpace):
    """Create InvertedResidual SearchSpace."""

    def constructor(self, inp, oup, stride, kernel=3, expand_ratio=1):
        """Construct InvertedResidual class.

        :param inp: input channel
        :param oup: output channel
        :param stride: stride
        :param kernel: kernel
        :param expand_ratio: channel increase multiplier
        """
        user_res_connect = stride == 1 and inp == oup
        if user_res_connect:
            conv = InvertedConv(inp, oup, stride, kernel, expand_ratio)
            self.inverted_residual = Add(conv)
        else:
            self.inverted_residual = InvertedConv(inp, oup, stride, kernel, expand_ratio)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class MergeCell(FineGrainedSpace):
    """Pass two inputs through ContextualCell, and aggregate their results."""

    def constructor(self, op_names, ctx_config, conn, inps, agg_size, ctx_cell, repeats=1,
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
        inp_1, inp_2 = inps
        self.op_1 = ctx_cell(op_names=op_names, config=ctx_config, inp=inp_1, repeats=repeats)
        self.op_2 = ctx_cell(op_names=op_names, config=ctx_config, inp=inp_2, repeats=repeats)
        self.agg = AggregateCell(size_1=inp_1, size_2=inp_2, agg_size=agg_size, concat=cell_concat)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class MicroDecoder(FineGrainedSpace):
    """Parent class for MicroDecoders."""

    def constructor(self, op_names, backbone_out_sizes, num_classes, config, agg_size=64, num_pools=4,
                    ctx_cell=ContextualCell_v1, aux_cell=False, sep_repeats=1, agg_concat=False, cell_concat=False,
                    **params):
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
        adapt = []
        for out_idx, size in enumerate(backbone_out_sizes):
            adapt.append(conv_bn_relu(inchannel=size, outchannel=agg_size,
                                      kernel_size=1, stride=1, padding=0, affine=True))
            backbone_out_sizes[out_idx] = agg_size
        if sys.version_info[0] < 3:
            backbone_out_sizes = list(backbone_out_sizes)
        else:
            backbone_out_sizes = backbone_out_sizes.copy()
        self.adapt = Process_list(*tuple(adapt))
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
                                   inps=(backbone_out_sizes[ind_1], backbone_out_sizes[ind_2]),
                                   agg_size=agg_size,
                                   ctx_cell=ctx_cell, repeats=sep_repeats,
                                   cell_concat=cell_concat))
            cell_concat = False
            collect_inds.append(block_idx + num_pools)
            backbone_out_sizes.append(agg_size)
            # for description
        self.block = Process_list(*tuple(cells), out_list=conns)
        self.upsample = MicroDecoder_Upsample(collect_inds=collect_inds, agg_concat=agg_concat)
        self.pre_clf = conv_bn_relu(inchannel=agg_size * (len(collect_inds) if agg_concat else 1),
                                    outchannel=agg_size, kernel_size=1, stride=1, padding=0)
        self.conv_clf = conv3x3(inchannel=agg_size, outchannel=num_classes, stride=1, bias=True)
