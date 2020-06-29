# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ResNet models for quantization."""
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks import NetTypes, NetworkFactory
from ..blocks.resnet_block import BasicBlock

__all__ = ['QuantResNet']


def _weights_init(m):
    """Random initialization of weights.

    :param m: pytorch model
    :type m: nn.Module
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


@NetworkFactory.register(NetTypes.BACKBONE)
class QuantResNet(Network):
    """QuantResNet class.

    :param descript: network description
    :type descript: dict
    """

    def __init__(self, descript):
        super(QuantResNet, self).__init__()
        self.net_desc = descript
        self.in_planes = 16
        block = BasicBlock
        num_blocks = descript.get('num_blocks', [3, 3, 3])
        num_classes = descript.get('num_classes', 10)
        self.nbit_w_list = descript.get('nbit_w_list', None)
        self.nbit_a_list = descript.get('nbit_a_list', None)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _init_weights(self):
        """Initialize the weights."""
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Construct layers in one stage of ResNet.

        :param block: type of block
        :type block: class
        :param planes: the number of channels
        :type planes: int
        :param num_blocks: the number of blocks
        :type num_blocks: int
        :param stride: stride of convolution
        :type stride: int
        :return: group of layers
        :rtype: nn.Sequential
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward.

        :param x: input
        :type x: Tensor
        :return: output
        :rtype: Tensor
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
