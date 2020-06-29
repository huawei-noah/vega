# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Efficient residual dense models for super-resolution."""
import torch
import torch.nn as nn
import logging
import math
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory


def channel_shuffle(x, groups):
    """Shuffle the channel of features.

    :param x: feature maps
    :type x: tensor
    :param groups: group number of channels
    :type groups: int
    :return: shuffled feature map
    :rtype: tensor
    """
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x


class RDB_Conv(nn.Module):
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
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1,
                      groups=self.congroup),
            nn.ReLU()
        ])

    def forward(self, x):
        """Forward function.

        :param x: input tensor
        :type x: tensor
        :return: the output of block
        :rtype: tensor
        """
        out = self.conv(channel_shuffle(x, groups=self.shgroup))
        return torch.cat((x, out), 1)


class Group_RDB(nn.Module):
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
            self.InConv = nn.Conv2d(self.InChan, self.G, 1, padding=0, stride=1)
        if self.OutChan != self.G and self.OutChan != self.InChan:
            self.OutConv = nn.Conv2d(self.InChan, self.OutChan, 1, padding=0, stride=1)

        convs = []
        for c in range(self.C):
            convs.append(RDB_Conv((c + 1) * self.G, self.G, c + 1,
                                  min(4, 2 ** int(math.log(c + 1, 2)))))
        self.convs = nn.Sequential(*convs)

        self.LFF = nn.Conv2d((self.C + 1) * self.G, self.OutChan, 1, padding=0,
                             stride=1)

    def forward(self, x):
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
            x_inter = self.LFF(self.convs(x))
        if self.OutChan == self.InChan:
            x_return = x + x_inter
        elif self.OutChan == self.G:
            x_return = x_InC + x_inter
        else:
            x_return = self.OutConv(x) + x_inter
        return x_return


class Shrink_RDB(nn.Module):
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
            self.InConv = nn.Conv2d(self.InChan, self.G, 1, padding=0, stride=1)
        if self.OutChan != self.G and self.OutChan != self.InChan:
            self.OutConv = nn.Conv2d(self.InChan, self.OutChan, 1, padding=0,
                                     stride=1)
        self.Convs = nn.ModuleList()
        self.ShrinkConv = nn.ModuleList()
        for i in range(self.C):
            self.Convs.append(nn.Sequential(*[
                nn.Conv2d(self.G, self.G, kSize, padding=(kSize - 1) // 2,
                          stride=1), nn.ReLU()]))
            if i == (self.C - 1):
                self.ShrinkConv.append(
                    nn.Conv2d((2 + i) * self.G, self.OutChan, 1, padding=0,
                              stride=1))
            else:
                self.ShrinkConv.append(
                    nn.Conv2d((2 + i) * self.G, self.G, 1, padding=0, stride=1))

    def forward(self, x):
        """Forward function.

        :param x: input tensor
        :type x: tensor
        :return: the output of block
        :rtype: tensor
        """
        if self.InChan != self.G:
            x_InC = self.InConv(x)
            x_inter = self.Convs[0](x_InC)
            x_conc = torch.cat((x_InC, x_inter), 1)
            x_in = self.ShrinkConv[0](x_conc)
        else:
            x_inter = self.Convs[0](x)
            x_conc = torch.cat((x, x_inter), 1)
            x_in = self.ShrinkConv[0](x_conc)
        for i in range(1, self.C):
            x_inter = self.Convs[i](x_in)
            x_conc = torch.cat((x_conc, x_inter), 1)
            x_in = self.ShrinkConv[i](x_conc)
        if self.OutChan == self.InChan:
            x_return = x + x_in
        elif self.OutChan == self.G:
            x_return = x_InC + x_in
        else:
            x_return = self.OutConv(x) + x_in
        return x_return


class Cont_RDB(nn.Module):
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
            self.InConv = nn.Conv2d(self.InChan, self.G, 1, padding=0, stride=1)
        if self.OutChan != self.G and self.OutChan != self.InChan:
            self.OutConv = nn.Conv2d(self.InChan, self.OutChan, 1, padding=0,
                                     stride=1)

        self.pool = nn.AvgPool2d(2, 2)
        self.shup = nn.PixelShuffle(2)
        self.Convs = nn.ModuleList()
        self.ShrinkConv = nn.ModuleList()
        for i in range(self.C):
            self.Convs.append(nn.Sequential(*[
                nn.Conv2d(self.G, self.G, kSize, padding=(kSize - 1) // 2,
                          stride=1), nn.ReLU()]))
            if i < (self.C - 1):
                self.ShrinkConv.append(
                    nn.Conv2d((2 + i) * self.G, self.G, 1, padding=0, stride=1))
            else:
                self.ShrinkConv.append(
                    nn.Conv2d(int((2 + i) * self.G / 4), self.OutChan, 1,
                              padding=0, stride=1))

    def forward(self, x):
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
            x_in = self.pool(x)
        x_conc = x_in
        for i in range(0, self.C):
            x_inter = self.Convs[i](x_in)
            x_inter = self.Convs[i](x_inter)
            x_inter = self.Convs[i](x_inter)
            x_conc = torch.cat((x_conc, x_inter), 1)
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


@NetworkFactory.register(NetTypes.ESRBODY)
class ESRN(Network):
    """Efficient super-resolution networks construction."""

    def __init__(self, net_desc):
        """Construct the ESRN class.

        :param net_desc: config of the searched structure
        :type net_desc: list
        """
        super(ESRN, self).__init__()
        logging.info("start init ESRN")
        self.desc = net_desc
        self.arch = net_desc.architecture
        self.D = len(self.arch)
        r = net_desc.scale
        G0 = net_desc.G0
        kSize = 3
        n_colors = 3
        self.SFENet1 = nn.Conv2d(n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        self.ERDBs = nn.ModuleList()
        b_in_chan = G0
        b_out_chan = 0
        Conc_all = 0
        for i in range(self.D):
            name = self.arch[i]
            key = name.split('_')
            if i > 0:
                b_in_chan = b_out_chan
            b_conv_num = int(key[1])
            b_grow_rat = int(key[2])
            b_out_chan = int(key[3])
            Conc_all += b_out_chan
            if key[0] == 'S':
                self.ERDBs.append(Shrink_RDB(InChannel=b_in_chan,
                                             OutChannel=b_out_chan,
                                             growRate=b_grow_rat,
                                             nConvLayers=b_conv_num))
            elif key[0] == 'G':
                self.ERDBs.append(Group_RDB(InChannel=b_in_chan,
                                            OutChannel=b_out_chan,
                                            growRate=b_grow_rat,
                                            nConvLayers=b_conv_num))
            elif key[0] == 'C':
                self.ERDBs.append(Cont_RDB(InChannel=b_in_chan,
                                           OutChannel=b_out_chan,
                                           growRate=b_grow_rat,
                                           nConvLayers=b_conv_num))
            else:
                logging.info('Wrong Block Type')

        self.GFF = nn.Sequential(*[
            nn.Conv2d(Conc_all, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G0 * 3, kSize, padding=(kSize - 1) // 2,
                          stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(int(G0 * 3 / 4), n_colors, kSize,
                          padding=(kSize - 1) // 2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G0 * 4, kSize, padding=(kSize - 1) // 2,
                          stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G0, G0 * 4, kSize, padding=(kSize - 1) // 2,
                          stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G0, n_colors, kSize, padding=(kSize - 1) // 2,
                          stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        """Calculate the output of the model.

        :param x: input tensor
        :type x: tensor
        :return: output tensor of the model
        :rtype: tensor
        """
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        ERDBs_out = []
        for i in range(self.D):
            x = self.ERDBs[i](x)
            ERDBs_out.append(x)

        x = self.GFF(torch.cat(ERDBs_out, 1))
        x += f__1

        return self.UPNet(x)
