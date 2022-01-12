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

"""Efficient residual dense models for super-resolution."""
import math
import logging
from vega.modules.module import Module
from vega.modules.operators import ops
from vega.modules.connections import Sequential
from vega.common.class_factory import ClassType, ClassFactory


def channel_shuffle(x, groups):
    """Shuffle the channel of features.

    :param x: feature maps
    :type x: tensor
    :param groups: group number of channels
    :type groups: int
    :return: shuffled feature map
    :rtype: tensor
    """
    batchsize, num_channels, height, width = ops.get_shape(x)
    channels_per_group = num_channels // groups
    x = ops.View([batchsize, groups, channels_per_group, height, width])(x)
    x = ops.Transpose(1, 2)(x)
    x = ops.View([batchsize, num_channels, height, width])(x)
    return x


class RDB_Conv(Module):
    """Convolution operation of efficient residual dense block with shuffle and group."""

    def __init__(self, inChannels, growRate, sh_groups, conv_groups, kSize=3):
        """Initialize Block.

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
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.shgroup = sh_groups
        self.congroup = conv_groups
        self.conv = Sequential(
            ops.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1,
                       groups=self.congroup),
            ops.Relu()
        )

    def call(self, x):
        """Forward function.

        :param x: input tensor
        :type x: tensor
        :return: the output of block
        :rtype: tensor
        """
        if self.data_format == "channels_first":
            out = self.conv(channel_shuffle(x, groups=self.shgroup))
        else:
            x = ops.Permute([0, 3, 1, 2])(x)
            out = self.conv(channel_shuffle(x, groups=self.shgroup))
            x = ops.Permute([0, 2, 3, 1])(x)
            out = ops.Permute([0, 2, 3, 1])(out)
        return ops.concat((x, out))


class Group_RDB(Module):
    """Group residual dense block."""

    def __init__(self, InChannel, OutChannel, growRate, nConvLayers, kSize=3):
        """Initialize Block.

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
        super(Group_RDB, self).__init__()
        self.InChan = InChannel
        self.OutChan = OutChannel
        self.G = growRate
        self.C = nConvLayers
        if self.InChan != self.G:
            self.InConv = ops.Conv2d(
                self.InChan, self.G, 1, padding=0, stride=1)
        if self.OutChan != self.G and self.OutChan != self.InChan:
            self.OutConv = ops.Conv2d(
                self.InChan, self.OutChan, 1, padding=0, stride=1)

        convs = []
        for c in range(self.C):
            convs.append(RDB_Conv((c + 1) * self.G, self.G, c + 1,
                                  min(4, 2 ** int(math.log(c + 1, 2)))))
        self.convs = Sequential(*convs)

        self.LFF = ops.Conv2d((self.C + 1) * self.G, self.OutChan, 1, padding=0,
                              stride=1)

    def call(self, x):
        """Forward function.

        :param x: input tensor
        :type x: tensor
        :return: the output of block
        :rtype: tensor
        """
        if self.InChan != self.G:
            x_InC = self.InConv(x)
            x_inter = self.LFF(self.convs(x_InC))
        else:
            x_InC = None
            x_inter = self.LFF(self.convs(x))
        if self.OutChan == self.InChan:
            x_return = x + x_inter
        elif self.OutChan == self.G:
            x_return = x_InC + x_inter
        else:
            x_return = self.OutConv(x) + x_inter
        return x_return


class Shrink_RDB(Module):
    """Shrink residual dense block."""

    def __init__(self, InChannel, OutChannel, growRate, nConvLayers, kSize=3):
        """Initialize Block.

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
        super(Shrink_RDB, self).__init__()
        self.InChan = InChannel
        self.OutChan = OutChannel
        self.G = growRate
        self.C = nConvLayers
        if self.InChan != self.G:
            self.InConv = ops.Conv2d(
                self.InChan, self.G, 1, padding=0, stride=1)
        if self.OutChan != self.G and self.OutChan != self.InChan:
            self.OutConv = ops.Conv2d(self.InChan, self.OutChan, 1, padding=0,
                                      stride=1)
        self.Convs = ops.MoudleList()
        self.ShrinkConv = ops.MoudleList()
        for i in range(self.C):
            self.Convs.append(Sequential(
                ops.Conv2d(self.G, self.G, kSize, padding=(kSize - 1) // 2,
                           stride=1), ops.Relu()))
            if i == (self.C - 1):
                self.ShrinkConv.append(
                    ops.Conv2d((2 + i) * self.G, self.OutChan, 1, padding=0,
                               stride=1))
            else:
                self.ShrinkConv.append(
                    ops.Conv2d((2 + i) * self.G, self.G, 1, padding=0, stride=1))

    def call(self, x):
        """Forward function.

        :param x: input tensor
        :type x: tensor
        :return: the output of block
        :rtype: tensor
        """
        if self.InChan != self.G:
            x_InC = self.InConv(x)
            x_inter = self.Convs[0](x_InC)
            x_conc = ops.concat((x_InC, x_inter))
            x_in = self.ShrinkConv[0](x_conc)
        else:
            x_InC = None
            x_inter = self.Convs[0](x)
            x_conc = ops.concat((x, x_inter))
            x_in = self.ShrinkConv[0](x_conc)
        for i in range(1, self.C):
            x_inter = self.Convs[i](x_in)
            x_conc = ops.concat((x_conc, x_inter))
            x_in = self.ShrinkConv[i](x_conc)
        if self.OutChan == self.InChan:
            x_return = x + x_in
        elif self.OutChan == self.G:
            x_return = x_InC + x_in
        else:
            x_return = self.OutConv(x) + x_in
        return x_return


class Cont_RDB(Module):
    """Contextual residual dense block."""

    def __init__(self, InChannel, OutChannel, growRate, nConvLayers, kSize=3):
        """Initialize Block.

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
        super(Cont_RDB, self).__init__()
        self.InChan = InChannel
        self.OutChan = OutChannel
        self.G = growRate
        self.C = nConvLayers
        if self.InChan != self.G:
            self.InConv = ops.Conv2d(
                self.InChan, self.G, 1, padding=0, stride=1)
        if self.OutChan != self.G and self.OutChan != self.InChan:
            self.OutConv = ops.Conv2d(
                self.InChan, self.OutChan, 1, padding=0, stride=1)
        self.pool = ops.AvgPool2d(2, 2)
        self.shup = ops.PixelShuffle(2)
        self.Convs = ops.MoudleList()
        self.ShrinkConv = ops.MoudleList()
        for i in range(self.C):
            self.Convs.append(Sequential(
                ops.Conv2d(self.G, self.G, kSize, padding=(kSize - 1) // 2,
                           stride=1), ops.Relu()))
            if i < (self.C - 1):
                self.ShrinkConv.append(ops.Conv2d(
                    (2 + i) * self.G, self.G, 1, padding=0, stride=1))
            else:
                self.ShrinkConv.append(
                    ops.Conv2d(int((2 + i) * self.G / 4), self.OutChan, 1,
                               padding=0, stride=1))

    def call(self, x):
        """Forward function.

        :param x: input tensor
        :type x: tensor
        :return: the output of block
        :rtype: tensor
        """
        if self.InChan != self.G:
            x_InC = self.InConv(x)
            x_in = self.pool(x_InC)
        else:
            x_InC = None
            x_in = self.pool(x)
        x_conc = x_in
        for i in range(0, self.C):
            x_inter = self.Convs[i](x_in)
            x_inter = self.Convs[i](x_inter)
            x_inter = self.Convs[i](x_inter)
            x_conc = ops.concat((x_conc, x_inter))
            if i == (self.C - 1):
                x_conc = self.shup(x_conc)
                x_in = self.ShrinkConv[i](x_conc)
            else:
                x_in = self.ShrinkConv[i](x_conc)
        if self.OutChan == self.InChan:
            x_return = x + x_in
        elif self.OutChan == self.G:
            x_return = x_InC + x_in
        else:
            x_return = self.OutConv(x) + x_in
        return x_return


class ERDBLayer(Module):
    """Create ERDBLayer Searchspace."""

    def __init__(self, arch, G0, kSize):
        """Create ERDBLayer.

        :param arch: arch
        :type arch: dict
        :param G0: G0
        :type G0: G0
        :param kSize: kSize
        :type kSize: int
        """
        super(ERDBLayer, self).__init__()
        self.SFENet2 = ops.Conv2d(
            G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        b_in_chan = G0
        b_out_chan = 0
        Conc_all = 0
        ERDBs = ops.MoudleList()
        for i in range(len(arch)):
            name = arch[i]
            key = name.split('_')
            if i > 0:
                b_in_chan = b_out_chan
            b_conv_num = int(key[1])
            b_grow_rat = int(key[2])
            b_out_chan = int(key[3])
            Conc_all += b_out_chan
            if key[0] == 'S':
                ERDBs.append(Shrink_RDB(InChannel=b_in_chan,
                                        OutChannel=b_out_chan,
                                        growRate=b_grow_rat,
                                        nConvLayers=b_conv_num))
            elif key[0] == 'G':
                ERDBs.append(Group_RDB(InChannel=b_in_chan,
                                       OutChannel=b_out_chan,
                                       growRate=b_grow_rat,
                                       nConvLayers=b_conv_num))
            elif key[0] == 'C':
                ERDBs.append(Cont_RDB(InChannel=b_in_chan,
                                      OutChannel=b_out_chan,
                                      growRate=b_grow_rat,
                                      nConvLayers=b_conv_num))
        self.ERBD = ERDBs
        self.GFF = Sequential(
            ops.Conv2d(Conc_all, G0, 1, padding=0, stride=1),
            ops.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        )

    def call(self, inputs):
        """Calculate the output of the model.

        :param x: input tensor
        :type x: tensor
        :return: output tensor of the model
        :rtype: tensor
        """
        x = self.SFENet2(inputs)

        ERDBs_out = ()
        for net in self.ERBD:
            x = net(x)
            ERDBs_out += (x,)

        x = self.GFF(ops.concat(ERDBs_out))
        x += inputs
        return x


@ClassFactory.register(ClassType.NETWORK)
class ESRN(Module):
    """Efficient super-resolution networks construction."""

    def __init__(self, block_type, conv_num, growth_rate, type_prob, conv_prob, growth_prob,
                 G0, scale, code, architecture):
        """Construct the ESRN class.

        :param net_desc: config of the searched structure
        :type net_desc: list
        """
        super(ESRN, self).__init__()

        logging.info("start init ESRN")
        self.arch = architecture
        self.D = len(self.arch)
        r = scale
        G0 = G0
        kSize = 3
        n_colors = 3
        self.SFENet1 = ops.Conv2d(
            n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.ERDBLayer = ERDBLayer(architecture, G0, kSize)
        if r == 2 or r == 3:
            self.UPNet = Sequential(
                ops.Conv2d(G0, G0 * 3, kSize, padding=(kSize - 1) // 2,
                           stride=1),
                ops.PixelShuffle(r),
                ops.Conv2d(int(G0 * 3 / 4), n_colors, kSize,
                           padding=(kSize - 1) // 2, stride=1)
            )
        elif r == 4:
            self.UPNet = Sequential(
                ops.Conv2d(G0, G0 * 4, kSize, padding=(kSize - 1) // 2,
                           stride=1),
                ops.PixelShuffle(2),
                ops.Conv2d(G0, G0 * 4, kSize, padding=(kSize - 1) // 2,
                           stride=1),
                ops.PixelShuffle(2),
                ops.Conv2d(G0, n_colors, kSize, padding=(kSize - 1) // 2,
                           stride=1)
            )
        else:
            raise ValueError("scale must be 2 or 3 or 4.")
