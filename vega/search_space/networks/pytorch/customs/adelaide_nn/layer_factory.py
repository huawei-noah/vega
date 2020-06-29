# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Different custom layers."""
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, bias=False, dilation=1):
    """Create Convolution 3x3.

    :param in_planes: input channels
    :param out_planes: output channels
    :param stride: stride of the convolution
    :param bias: whether bias is contained
    :param dilation: dilation of the convolution
    :return: a convolution module
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=bias)


def conv5x5(in_planes, out_planes, stride=1, bias=False, dilation=1):
    """Create Convolution 5x5.

    :param in_planes: input channels
    :param out_planes: output channels
    :param stride: stride of the convolution
    :param bias: whether bias is contained
    :param dilation: dilation of the convolution
    :return: a convolution module
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, dilation=dilation, bias=bias)


def conv7x7(in_planes, out_planes, stride=1, bias=False, dilation=1):
    """Create Convolution 7x7.

    :param in_planes: input channels
    :param out_planes: output channels
    :param stride: stride of the convolution
    :param bias: whether bias is contained
    :param dilation: dilation of the convolution
    :return: a convolution module
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, dilation=dilation, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """Create Convolution 1x1.

    :param in_planes: input channels
    :param out_planes: output channels
    :param stride: stride of the convolution
    :param bias: whether bias is contained
    :return: a convolution module
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


OPS = {
    'none': lambda C, stride, affine, repeats=1: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine, repeats=1: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine, repeats=1: nn.MaxPool2d(
        3, stride=stride, padding=1),
    'global_average_pool': lambda C, stride, affine, repeats=1: GAPConv1x1(C, C),
    'skip_connect': lambda C, stride, affine, repeats=1: Identity() if stride == 1 else FactorizedReduce(
        C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine, repeats=1: SepConv(C, C, 3, stride, 1, affine=affine, repeats=repeats),
    'sep_conv_5x5': lambda C, stride, affine, repeats=1: SepConv(C, C, 5, stride, 2, affine=affine, repeats=repeats),
    'sep_conv_7x7': lambda C, stride, affine, repeats=1: SepConv(C, C, 7, stride, 3, affine=affine, repeats=repeats),
    'dil_conv_3x3': lambda C, stride, affine, repeats=1: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine, repeats=1: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine, repeats=1: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)),
    'conv1x1': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv1x1(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv5x5': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv5x5(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv7x7': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv7x7(C, C, stride=stride),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3_dil2': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride, dilation=2),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3_dil3': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride, dilation=3),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'conv3x3_dil12': lambda C, stride, affine, repeats=1: nn.Sequential(
        conv3x3(C, C, stride=stride, dilation=12),
        nn.BatchNorm2d(C, affine=affine),
        nn.ReLU(inplace=False)),
    'sep_conv_3x3_dil3': lambda C, stride, affine, repeats=1: SepConv(
        C, C, 3, stride, 3, affine=affine, dilation=3, repeats=repeats),
    'sep_conv_5x5_dil6': lambda C, stride, affine, repeats=1: SepConv(
        C, C, 5, stride, 12, affine=affine, dilation=6, repeats=repeats)
}


def conv_bn_relu(C_in, C_out, kernel_size, stride, padding, affine=True):
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
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                  bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(inplace=False),
    )


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


class Identity(nn.Module):
    """Identity block."""

    def __init__(self):
        """Construct Identity class."""
        super(Identity, self).__init__()

    def forward(self, x):
        """Do an inference on Identity.

        :param x: input tensor
        :return: output tensor
        """
        return x


class Zero(nn.Module):
    """Zero block."""

    def __init__(self, stride):
        """Construct Zero class.

        :param stride: stride of the output
        """
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        """Do an inference on Zero.

        :param x: input tensor
        :return: output tensor
        """
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
    """Factorized reduce block."""

    def __init__(self, C_in, C_out, affine=True):
        """Construct FactorizedReduce class.

        :param C_in: input channel
        :param C_out: output channel
        :param affine: whether to use affine in BN
        """
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        """Do an inference on FactorizedReduce.

        :param x: input tensor
        :return: output tensor
        """
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
