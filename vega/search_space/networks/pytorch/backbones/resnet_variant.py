# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ResNet models ResNetVariant."""
import torch.nn as nn
import torch.nn.init as init
import logging
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory
from ..blocks.resnet_block import BasicBlock, BottleneckBlock, SmallInputInitialBlock, InitialBlock


def _weights_init(m):
    """Random initialization of weights."""
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


@NetworkFactory.register(NetTypes.BACKBONE)
class ResNetVariant(Network):
    """ResNetVariant.

    :param net_desc: Description of ResNetVariant.
    :type net_desc: NetworkDesc

    """

    _block_setting = {18: ('BasicBlock', 8),
                      34: ('BasicBlock', 16),
                      50: ('BottleneckBlock', 16),
                      101: ('BottleneckBlock', 33)}

    def __init__(self, net_desc):
        """Init ResNetVariant."""
        super(ResNetVariant, self).__init__()
        self.net_desc = net_desc
        logging.info("start init ResNetVariant")
        self.small_input = True
        self.base_depth = int(net_desc.base_depth)
        self.base_channel = int(net_desc.base_channel)
        self.doublechannel = net_desc.doublechannel
        self.downsample = net_desc.downsample

        self.num_block = self._block_setting[self.base_depth][1]
        block_name = self._block_setting[self.base_depth][0]

        if block_name == 'BasicBlock':
            self.block = BasicBlock
        else:
            self.block = BottleneckBlock

        self.out_channel = self.base_channel * \
            2 ** sum(self.doublechannel) * self.block.expansion
        if self.small_input:
            self.init_block = SmallInputInitialBlock(self.base_channel)
        else:
            self.init_block = InitialBlock(self.base_channel)
        blocks = self._make_resolution_block(self.base_channel,
                                             self.out_channel,
                                             self.num_block,
                                             self.downsample,
                                             self.doublechannel)
        logging.info("start init ResNetVariant blocks")
        self.blocks = nn.Sequential(*blocks)
        self.apply(_weights_init)
        logging.info("finished init ResNetVariant blocks")

    def _make_resolution_block(self, in_planes, out_planes, num_blocks,
                               downsample, doublechannel):
        """Build resolution blocks.

        :param in_planes: input channel.
        :type in_planes: int
        :param out_planes: output channel.
        :type out_planes: int
        :param num_blocks: number of blocks.
        :type num_blocks: int
        :param downsample: downsample position, 1 for use downsample, 0 for not.
        :type downsample: list of (0,1)
        :param doublechannel: doublechannel position, 1 for use downsample, 0 for not.
        :type doublechannel: list of (0,1)

        """
        _msg = "_make_resolution_block(in_planes={}, out_planes={}, num_blocks={}, downsample={}, doublechannel={})"
        logging.debug(_msg.format(in_planes, out_planes, num_blocks, downsample, doublechannel))
        blocks = []
        i_in_planes = in_planes
        i_out_planes = in_planes
        for idx in range(len(doublechannel)):
            logging.debug("_make_resolution_block:{}".format(idx))
            i_out_planes = i_out_planes if doublechannel[idx] == 0 else i_out_planes * 2
            blocks.append(
                self.block(
                    in_planes=i_in_planes,
                    out_planes=i_out_planes,
                    stride=1 if downsample[idx] == 0 else 2,
                ),
            )
            i_in_planes = i_out_planes * self.block.expansion
        return blocks

    def forward(self, x):
        """Forward function of ResNet."""
        out = self.init_block(x)
        return self.blocks(out)

    @property
    def input_shape(self):
        """Get the model input tensor shape."""
        if self.small_input:
            return (3, 32, 32)
        else:
            return (3, 224, 224)

    @property
    def output_shape(self):
        """Get the model output tensor shape."""
        if self.small_input:
            return (3, self.out_channel, 32, 32)
        else:
            return (3, self.out_channel, 224, 224)

    @property
    def model_layers(self):
        """Get the model layers."""
        return self.base_depth
