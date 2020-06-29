# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ResNet models for pruning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import functools
import logging
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory


def initialize_weights(net_l, scale=1.0):
    """Init parameters using kaiming_normal_ method.

    :param net_l: parameter or list of parameters
    :param scale: rescale ratio of parameters
    """
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class MeanShift(nn.Conv2d):
    """Subtract or add rgb_mean to the image."""

    def __init__(self, rgb_range, rgb_mean, rgb_std=(1.0, 1.0, 1.0), sign=-1):
        """Construct the class MeanShift.

        :param rgb_range: range of tensor, usually 1.0 or 255.0
        :param rgb_mean: mean of rgb value
        :param rgb_std: std of rgb value
        :param sign: -1 for subtract, 1 for add
        """
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


def conv(in_channel, out_channel, kernel_size=3, padding=None):
    """Make a convolution layer with dilation 1, groups 1.

    :param in_channel: number of input channels
    :param out_channel: number of output channels
    :param kernel_size: kernel size
    :param padding: padding, Setting None to be same
    :return: convolution layer as set
    """
    if padding is None:
        padding = kernel_size // 2

    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                     padding=padding)


class ResidualBlock(nn.Module):
    """Basic block with channel number and kernel size as variable."""

    def __init__(self, kernel_size=3, base_channel=64):
        """
        Construct the ResidualBlock class.

        :param kernel_size: kernel size of conv layers
        :param base_channel: number of input (and output) channels
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = conv(base_channel, base_channel, kernel_size, padding=kernel_size // 2)
        self.conv2 = conv(base_channel, base_channel, kernel_size, padding=(kernel_size - 1) // 2)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        """Calculate the output of the model.

        :param x: input tensor
        :return: output tensor of the model
        """
        y = F.leaky_relu(self.conv1(x), inplace=True)
        y = self.conv2(y)

        return x + y


NAME_BLOCKS = {
    'res2': functools.partial(ResidualBlock, kernel_size=2),
    'res3': functools.partial(ResidualBlock, kernel_size=3)
}


class ChannelIncreaseBlock(nn.Module):
    """Channel increase block, which passes several blocks, and concat the result on channel dim."""

    def __init__(self, blocks, base_channel):
        """Construct the class ChannelIncreaseBlock.

        :param blocks: list of string of the blocks
        :param base_channel: number of input channels
        """
        super(ChannelIncreaseBlock, self).__init__()
        self.layers = nn.ModuleList(
            [NAME_BLOCKS[block_name](base_channel=base_channel) for block_name in blocks])

    def forward(self, x):
        """Calculate the output of the model.

        :param x: input tensor
        :return: output tensor of the model
        """
        out = []
        for block in self.layers:
            x = block(x)
            out.append(x)

        return torch.cat(out, 1)


@NetworkFactory.register(NetTypes.CUSTOM)
class MtMSR(Network):
    """Search space of MtM-NAS."""

    def __init__(self, net_desc):
        """Construct the MtMSR class.

        :param net_desc: config of the searched structure
        """
        super(MtMSR, self).__init__()
        logging.info("start init MTMSR")
        logging.info("MtMSR desc:", net_desc)
        self.desc = net_desc
        current_channel = net_desc.in_channel
        out_channel = net_desc.out_channel
        upscale = net_desc.upscale
        rgb_mean = net_desc.rgb_mean
        layers = list()
        for block_name in net_desc.blocks:
            if isinstance(block_name, list):
                layers.append(ChannelIncreaseBlock(block_name, current_channel))
                current_channel *= len(block_name)
            else:
                layers.append(NAME_BLOCKS[block_name](base_channel=current_channel))
        layers.extend([
            conv(current_channel, out_channel * upscale ** 2),
            nn.PixelShuffle(upscale)
        ])
        initialize_weights(layers[-2], 0.1)
        self.sub_mean = MeanShift(1.0, rgb_mean)
        self.add_mean = MeanShift(1.0, rgb_mean, sign=1)
        self.body = nn.Sequential(*layers)
        self.upsample = nn.Upsample(scale_factor=upscale, mode="bilinear", align_corners=False)

    def forward(self, x):
        """Calculate the output of the model.

        :param x: input tensor
        :return: output tensor of the model
        """
        x = self.sub_mean(x)
        y = self.body(x)
        x = self.upsample(x)
        y = x + y
        y = self.add_mean(y)
        return y
