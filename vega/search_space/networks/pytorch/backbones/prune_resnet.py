# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ResNet models for pruning."""
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks import NetTypes, NetworkFactory
from ..blocks.resnet_block import PruneBasicBlock


def _weights_init(m):
    """Random initialization of weights."""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


@NetworkFactory.register(NetTypes.BACKBONE)
class PruneResNet(Network):
    """PruneResNet.

    :param descript: network desc
    :type descript: dict
    """

    def __init__(self, descript):
        """Init PruneResNet."""
        super(PruneResNet, self).__init__()
        self.net_desc = descript
        block = descript.get('block', 'PruneBasicBlock')
        if block == 'PruneBasicBlock':
            self.block = eval(block)
        else:
            raise TypeError('Do not have this block type: {}'.format(block))
        self.encoding = descript.get('encoding')
        self.chn = descript.get('chn')
        self.chn_node = descript.get('chn_node')
        self.chn_mask = descript.get('chn_mask', None)
        self.chn_node_mask = descript.get('chn_node_mask', None)
        num_blocks = descript.get('num_blocks', [3, 3, 3])
        num_classes = descript.get('num_classes', 10)
        self.in_planes = self.chn_node[0]
        self.conv1 = nn.Conv2d(
            3, self.chn_node[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.chn_node[0])
        self.layer1 = self._make_layer(
            self.block, self.chn_node[1], self.chn[0:3], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            self.block, self.chn_node[2], self.chn[3:6], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            self.block, self.chn_node[3], self.chn[6:9], num_blocks[2], stride=2)
        self.linear = nn.Linear(self.chn_node[3], num_classes)

    def _init_weights(self):
        """Init weights."""
        self.apply(_weights_init)

    def _make_layer(self, block, planes, inner_planes, num_blocks, stride):
        """Construct layers in one stage of ResNet.

        :param planes: output channel number in every basicblock
        :type planes: int
        :param inner_planes: middle layer channel number in every basicblock
        :type inner_planes: int
        :param num_blocks: block number in every stage
        :type num_blocks: int
        :param stride: stride in convolution operation
        :type stride: int
        """
        idx = 0
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,
                                inner_planes[idx], stride))
            self.in_planes = planes * block.expansion
            idx += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function of ResNet."""
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
