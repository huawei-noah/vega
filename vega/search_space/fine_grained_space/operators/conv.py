# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torch operators."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vega.core.common.class_factory import ClassType, ClassFactory


@ClassFactory.register(ClassType.SEARCH_SPACE)
def conv3x3(inchannel, outchannel, groups=1, stride=1, bias=False):
    """Create conv3x3 layer.

    :param inchannel: input channel.
    :type inchannel: int
    :param outchannel: output channel.
    :type outchannel: int
    :param stride: the number to jump, default 1
    :type stride: int
    """
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=bias)


@ClassFactory.register(ClassType.SEARCH_SPACE)
def conv1X1(inchannel, outchannel, stride=1):
    """Create conv1X1 layer.

    :param inchannel: input channel.
    :type inchannel: int
    :param outchannel: output channel.
    :type outchannel: int
    :param stride: the number to jump, default 1
    :type stride: int
    """
    return nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False)


@ClassFactory.register(ClassType.SEARCH_SPACE)
def conv5x5(inchannel, outchannel, stride=1, bias=False, dilation=1):
    """Create Convolution 5x5.

    :param in_planes: input channels
    :param out_planes: output channels
    :param stride: stride of the convolution
    :param bias: whether bias is contained
    :param dilation: dilation of the convolution
    :return: a convolution module
    """
    return nn.Conv2d(inchannel, outchannel, kernel_size=5, stride=stride,
                     padding=2, dilation=dilation, bias=bias)


@ClassFactory.register(ClassType.SEARCH_SPACE)
def conv7x7(inchannel, outchannel, stride=1, bias=False, dilation=1):
    """Create Convolution 7x7.

    :param in_planes: input channels
    :param out_planes: output channels
    :param stride: stride of the convolution
    :param bias: whether bias is contained
    :param dilation: dilation of the convolution
    :return: a convolution module
    """
    return nn.Conv2d(inchannel, outchannel, kernel_size=7, stride=stride,
                     padding=3, dilation=dilation, bias=bias)


@ClassFactory.register(ClassType.SEARCH_SPACE)
def conv_bn_relu6(inchannel, outchannel, kernel=3, stride=1):
    """Create conv1X1 layer.

    :param inchannel: input channel.
    :type inchannel: int
    :param outchannel: output channel.
    :type outchannel: int
    :param stride: the number to jump, default 1
    :type stride: int
    """
    return nn.Sequential(
        nn.Conv2d(inchannel, outchannel, kernel, stride, kernel // 2, bias=False),
        nn.BatchNorm2d(outchannel),
        nn.ReLU6(inplace=True)
    )


@ClassFactory.register(ClassType.SEARCH_SPACE)
def conv_bn_relu(inchannel, outchannel, kernel_size, stride, padding, affine=True):
    """Create group of Convolution + BN + Relu.

    :param C_in: input channel
    :param C_out: output channel
    :param kernel_size: kernel size of convolution layer
    :param stride: stride of convolution layer
    :param padding: padding of convolution layer
    :param affine: whether use affine in batchnorm
    :return: group of Convolution + BN + Relu
    """
    return nn.Sequential(
        nn.Conv2d(inchannel, outchannel, kernel_size, stride=stride, padding=padding,
                  bias=False),
        nn.BatchNorm2d(outchannel, affine=affine),
        nn.ReLU(inplace=False),
    )


@ClassFactory.register(ClassType.SEARCH_SPACE)
class ConvWS2d(nn.Conv2d):
    """Conv2d with weight standarlization.

    :param in_channels: input channels
    :param out_channels: output channels
    :param kernel_size: kernel size
    :param stride: stride
    :param padding: num of padding
    :param dilation: num of dilation
    :param groups: num of groups
    :param bias: bias
    :param eps: eps
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, eps=1e-5):
        """Init conv2d with weight standarlization."""
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def conv_ws_2d(self, input, weight, bias=None, stride=1, padding=0,
                   dilation=1, groups=1, eps=1e-5):
        """Conv2d with weight standarlization.

        :param input: input feature map
        :type input: torch.Tensor
        :param weight: weight of conv layer
        :type weight: torch.Tensor
        :param bias: bias
        :type bias: torch.Tensor
        :param stride: conv stride
        :type stride: int
        :param padding: num of padding
        :type padding: int
        :param dilation: num of dilation
        :type dilation: int
        :param groups: num of group
        :type groups: int
        :param eps: weight eps
        :type eps: float
        """
        c_in = weight.size(0)
        weight_flat = weight.view(c_in, -1)
        mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
        std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
        weight = (weight - mean) / (std + eps)
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    def forward(self, x):
        """Forward function of conv2d with weight standarlization."""
        return self.conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding,
                               self.dilation, self.groups, self.eps)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class ChannelShuffle(nn.Module):
    """Shuffle the channel of features.

    :param groups: group number of channels
    :type groups: int
    """

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Forward x."""
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Shrink_Conv(nn.Module):
    """Call torch.cat.

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

    def __init__(self, InChannel, OutChannel, growRate, nConvLayers, kSize):
        super(Shrink_Conv, self).__init__()
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
        """Forward x."""
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
        return x_in


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Cont_Conv(nn.Module):
    """Call torch.cat.

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

    def __init__(self, InChannel, OutChannel, growRate, nConvLayers, kSize=3):
        super(Cont_Conv, self).__init__()
        self.InChan = InChannel
        self.OutChan = OutChannel
        self.G = growRate
        self.C = nConvLayers
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
        """Forward x."""
        x_conc = x
        for i in range(0, self.C):
            x_inter = self.Convs[i](x)
            x_inter = self.Convs[i](x_inter)
            x_inter = self.Convs[i](x_inter)
            x_conc = torch.cat((x_conc, x_inter), 1)
            if i == (self.C - 1):
                x_conc = self.shup(x_conc)
                x_in = self.ShrinkConv[i](x_conc)
            else:
                x_in = self.ShrinkConv[i](x_conc)
        return x_in


@ClassFactory.register(ClassType.SEARCH_SPACE)
class GAPConv1x1(nn.Module):
    """Global Average Pooling + conv1x1."""

    def __init__(self, C_in, C_out):
        """Construct GAPConv1x1 class.

        :param C_in: input channel
        :param C_out: output channel
        """
        super(GAPConv1x1, self).__init__()
        if '0.2' in torch.__version__:
            # !!!!!!!!!!used for input size 448 with overall stride 32!!!!!!!!!!
            self.globalpool = nn.AvgPool2d(14)
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, stride=1, padding=0)

    def forward(self, x):
        """Do an inference on GAPConv1x1.

        :param x: input tensor
        :return: output tensor
        """
        size = x.size()[2:]
        if '0.2' in torch.__version__:
            out = self.globalpool(x)
        else:
            out = x.mean(2, keepdim=True).mean(3, keepdim=True)
        out = self.conv1x1(out)
        if '0.2' in torch.__version__:
            out = nn.Upsample(size=size, mode='bilinear')(out)
        else:
            out = nn.functional.interpolate(out, size=size, mode='bilinear', align_corners=False)
        return out


@ClassFactory.register(ClassType.SEARCH_SPACE)
class DilConv(nn.Module):
    """Separable dilated convolution block."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                 dilation, affine=True):
        """Construct DilConv class.

        :param C_in: input channel
        :param C_out: output channel
        :param kernel_size: kernel size of the first convolution layer
        :param stride: stride of the first convolution layer
        :param padding: padding of the first convolution layer
        :param dilation: dilation of the first convolution layer
        :param affine: whether use affine in BN
        """
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        """Do an inference on DilConv.

        :param x: input tensor
        :return: output tensor
        """
        return self.op(x)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class SepConv(nn.Module):
    """Separable convolution block with repeats."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1, affine=True, repeats=1):
        """Construct SepConv class.

        :param C_in: number of input channel
        :param C_out: number of output channel
        :param kernel_size: kernel size of the first conv
        :param stride: stride of the first conv
        :param padding: padding of the first conv
        :param dilation: dilation of the first conv
        :param affine: whether to use affine in BN
        :param repeats: number of repeat times
        """
        super(SepConv, self).__init__()

        def basic_op():
            return nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_in, affine=affine),
                nn.ReLU(inplace=False))

        self.op = nn.Sequential()
        for idx in range(repeats):
            self.op.add_module('sep_{}'.format(idx),
                               basic_op())

    def forward(self, x):
        """Do an inference on SepConv.

        :param x: input tensor
        :return: output tensor
        """
        return self.op(x)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class ReLUConvBN(nn.Module):
    """Class of ReLU + Conv + BN.

    :param C_in: input channel
    :type C_in: int
    :param C_out: output channel
    :type C_out: int
    :param kernel_size: kernel size of convolution
    :type kernel_size: int
    :param stride: stride of convolution
    :type stride: int
    :param padding: padding of convolution
    :type padding: int
    :param affine: whether to affine in bn, default True
    :type affine: bool
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        """Init ReLUConvBN."""
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        """Forward function fo ReLUConvBN."""
        return self.op(x)
