# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined darts operations."""
import torch
import torch.nn as nn
from .block import Block
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.pytorch.blocks.operations import *
from vega.search_space.networks import NetTypes, NetworkFactory


@NetworkFactory.register(NetTypes.BLOCK)
class none(Block):
    """Class of none.

    :param desc: description of none
    :type desc: Config
    """

    def __init__(self, desc):
        """Init none."""
        super(none, self).__init__()
        self.block = Zero(desc)


@NetworkFactory.register(NetTypes.BLOCK)
class avg_pool_3x3(Block):
    """Class of 3x3 average pooling.

    :param desc: description of avg_pool_3x3
    :type desc: Config
    """

    def __init__(self, desc):
        """Init avg_pool_3x3."""
        super(avg_pool_3x3, self).__init__()
        stride = desc.stride
        self.block = nn.AvgPool2d(
            3, stride=stride, padding=1, count_include_pad=False)


@NetworkFactory.register(NetTypes.BLOCK)
class max_pool_3x3(Block):
    """Class 3x3 max pooling.

    :param desc: description of max_pool_3x3
    :type desc: Config
    """

    def __init__(self, desc):
        """Init max_pool_3x3."""
        super(max_pool_3x3, self).__init__()
        stride = desc.stride
        self.block = nn.MaxPool2d(3, stride=stride, padding=1)


@NetworkFactory.register(NetTypes.BLOCK)
class skip_connect(Block):
    """Class of skip connect.

    :param desc: description of skip_connect
    :type desc: Config
    """

    def __init__(self, desc):
        """Init skip_connect."""
        super(skip_connect, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        if desc.stride == 1:
            self.block = Identity()
        else:
            self.block = FactorizedReduce(desc)


@NetworkFactory.register(NetTypes.BLOCK)
class sep_conv_3x3(Block):
    """Class of 3x3 separated convolution.

    :param desc: description of sep_conv_3x3
    :type desc: Config
    """

    def __init__(self, desc):
        """Init sep_conv_3x3."""
        super(sep_conv_3x3, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 3
        desc.padding = 1
        self.block = SeparatedConv(desc)


@NetworkFactory.register(NetTypes.BLOCK)
class sep_conv_5x5(Block):
    """Class of 5x5 separated convolution.

    :param desc: description of sep_conv_5x5
    :type desc: Config
    """

    def __init__(self, desc):
        """Init sep_conv_5x5."""
        super(sep_conv_5x5, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 5
        desc.padding = 2
        self.block = SeparatedConv(desc)


@NetworkFactory.register(NetTypes.BLOCK)
class sep_conv_7x7(Block):
    """Class of 7x7 separated convolution.

    :param desc: description of sep_conv_7x7
    :type desc: Config
    """

    def __init__(self, desc):
        """Init sep_conv_7x7."""
        super(sep_conv_7x7, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 7
        desc.padding = 3
        self.block = SeparatedConv(desc)


@NetworkFactory.register(NetTypes.BLOCK)
class dil_conv_3x3(Block):
    """Class of 3x3 dilation convolution.

    :param desc: description of dil_conv_3x3
    :type desc: Config
    """

    def __init__(self, desc):
        """Init dil_conv_3x3."""
        super(dil_conv_3x3, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 3
        desc.padding = 2
        desc.dilation = 2
        self.block = DilatedConv(desc)


@NetworkFactory.register(NetTypes.BLOCK)
class dil_conv_5x5(Block):
    """Class of 5x5 dilation convolution.

    :param desc: description of dil_conv_5x5
    :type desc: Config
    """

    def __init__(self, desc):
        """Init dil_conv_5x5."""
        super(dil_conv_5x5, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 5
        desc.padding = 4
        desc.dilation = 2
        self.block = DilatedConv(desc)


@NetworkFactory.register(NetTypes.BLOCK)
class conv_7x1_1x7(Block):
    """Class of 7x1 and 1x7 convolution.

    :param desc: description of conv_7x1_1x7
    :type desc: Config
    """

    def __init__(self, desc):
        """Init conv_7x1_1x7."""
        super(conv_7x1_1x7, self).__init__()
        stride = desc.stride
        channel_in = desc.C
        channel_out = desc.C
        affine = desc.affine
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_out, (1, 7), stride=(1, stride),
                      padding=(0, 3), bias=False),
            nn.Conv2d(channel_in, channel_out, (7, 1), stride=(stride, 1),
                      padding=(3, 0), bias=False),
            nn.BatchNorm2d(channel_out, affine=affine)
        )


@NetworkFactory.register(NetTypes.BLOCK)
class PreOneStem(Network):
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


@NetworkFactory.register(NetTypes.BLOCK)
class PreTwoStem(Network):
    """Class of two stems convolution.

    :param desc: description of PreTwoStem
    :type desc: Config
    """

    def __init__(self, desc):
        """Init PreTwoStem."""
        super(PreTwoStem, self).__init__()
        self._C = desc.C
        self.stems = nn.ModuleList()
        stem0 = nn.Sequential(
            nn.Conv2d(3, self._C // 2, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(self._C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self._C // 2, self._C, 3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(self._C),
        )
        self.stems += [stem0]
        stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self._C, self._C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self._C),
        )
        self.stems += [stem1]
        self.C_curr = self._C

    def forward(self, x):
        """Forward function of PreTwoStem."""
        out = [x]
        for stem in self.stems:
            out += [stem(out[-1])]
        return out[-2], out[-1]
