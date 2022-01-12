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

"""ResNet models for sr_ea."""
import logging
import functools
from vega.modules.module import Module
from vega.modules.operators import ops
from vega.common import ClassFactory, ClassType
from vega.modules.connections import Sequential, Add


def initialize_weights(net_l, scale=1.0):
    """Init parameters using kaiming_normal_ method.

    :param net_l: parameter or list of parameters
    :param scale: rescale ratio of parameters
    """
    if not isinstance(net_l, list):
        net_l = [net_l]
    for m in net_l:
        if m.__class__.__name__ == "Conv2d":
            m.initial(kernel_mode='he', bias_mode='zero', kernel_scale=0.1)


def conv(in_channel, out_channel, kernel_size=3, padding=None, sep=False):
    """Make a convolution layer with dilation 1, groups 1.

    :param in_channel: number of input channels
    :param out_channel: number of output channels
    :param kernel_size: kernel size
    :param padding: padding, Setting None to be same
    :return: convolution layer as set
    """
    if padding is None:
        padding = kernel_size // 2

    return ops.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, padding_mode="same")


class ResidualBlock(Module):
    """Basic block with channel number and kernel size as variable."""

    def __init__(self, kernel_size=3, base_channel=64):
        """
        Construct the ResidualBlock class.

        :param kernel_size: kernel size of conv layers
        :param base_channel: number of input (and output) channels
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = conv(base_channel, base_channel,
                          kernel_size, padding=kernel_size // 2)
        self.conv2 = conv(base_channel, base_channel,
                          kernel_size, padding=(kernel_size - 1) // 2)
        self.leaky_relu = ops.LeakyReLU(inplace=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def call(self, inputs, *args, **kwargs):
        """Calculate the output of the model.

        :param x: input tensor
        :return: output tensor of the model
        """
        x = inputs
        y = self.leaky_relu(self.conv1(x))
        y = self.conv2(y)

        return x + y


NAME_BLOCKS = {
    'res2': functools.partial(ResidualBlock, kernel_size=2),
    'res3': functools.partial(ResidualBlock, kernel_size=3)
}


class ChannelIncreaseBlock(Module):
    """Channel increase block, which passes several blocks, and concat the result on channel dim."""

    def __init__(self, blocks, base_channel):
        """Construct the class ChannelIncreaseBlock.

        :param blocks: list of string of the blocks
        :param base_channel: number of input channels
        """
        super(ChannelIncreaseBlock, self).__init__()
        self.layers = list()
        for block_name in blocks:
            self.layers.append(NAME_BLOCKS[block_name](
                base_channel=base_channel))
        self.layers = Sequential(*self.layers)
        self.blocks = self.layers.children() if isinstance(self.layers.children(), list) else list(
            self.layers.children())

    def call(self, inputs):
        """Calculate the output of the model.

        :param x: input tensor
        :return: output tensor of the model
        """
        out = ()
        x = inputs
        for block in self.blocks:
            x = block(x)
            out += (x,)

        return ops.concat(out)


@ClassFactory.register(ClassType.NETWORK)
class MtMSR(Module):
    """Search space of MtM-NAS."""

    def __init__(self, in_channel, out_channel, upscale, rgb_mean, blocks,
                 candidates, cib_range, method, code, block_range):
        """Construct the MtMSR class.

        :param net_desc: config of the searched structure
        """
        super(MtMSR, self).__init__()
        logging.info("start init MTMSR")
        current_channel = in_channel
        layers = list()
        for i, block_name in enumerate(blocks):
            if isinstance(block_name, list):
                layers.append(ChannelIncreaseBlock(block_name, current_channel))
                current_channel *= len(block_name)
            else:
                if block_name == "res2":
                    layers.append(ResidualBlock(
                        kernel_size=2, base_channel=current_channel))
                elif block_name == "res3":
                    layers.append(ResidualBlock(
                        kernel_size=3, base_channel=current_channel))
        layers.extend([
            conv(current_channel, out_channel * upscale ** 2),
            ops.PixelShuffle(upscale)
        ])
        initialize_weights(layers[-2], 0.1)
        self.sub_mean = ops.MeanShift(1.0, rgb_mean)
        body = Sequential(*layers)
        upsample = ops.InterpolateScale(scale_factor=upscale)
        self.add = Add(body, upsample)
        self.head = ops.MeanShift(1.0, rgb_mean, sign=1)
