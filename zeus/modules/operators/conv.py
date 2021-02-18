# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torch operators."""
from zeus.common import ClassType, ClassFactory
from zeus.modules.operators import ops


@ClassFactory.register(ClassType.NETWORK)
def conv3x3(inchannel, outchannel, groups=1, stride=1, bias=False, dilation=1):
    """Create conv3x3 layer."""
    return ops.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride,
                      padding=dilation, groups=groups, bias=bias, dilation=dilation)


@ClassFactory.register(ClassType.NETWORK)
def conv1X1(inchannel, outchannel, stride=1):
    """Create conv1X1 layer."""
    return ops.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False)


@ClassFactory.register(ClassType.NETWORK)
def conv5x5(inchannel, outchannel, stride=1, bias=False, dilation=1):
    """Create Convolution 5x5."""
    return ops.Conv2d(inchannel, outchannel, kernel_size=5, stride=stride,
                      padding=2, dilation=dilation, bias=bias)


@ClassFactory.register(ClassType.NETWORK)
def conv7x7(inchannel, outchannel, stride=1, bias=False, dilation=1):
    """Create Convolution 7x7."""
    return ops.Conv2d(inchannel, outchannel, kernel_size=7, stride=stride,
                      padding=3, dilation=dilation, bias=bias)


@ClassFactory.register(ClassType.NETWORK)
def conv_bn_relu6(C_in, C_out, kernel_size=3, stride=1, padding=0, affine=True):
    """Create group of Convolution + BN + Relu6 function."""
    return ConvBnRelu(C_in, C_out, kernel_size, stride, padding, affine=affine, use_relu6=True)


@ClassFactory.register(ClassType.NETWORK)
def conv_bn_relu(C_in, C_out, kernel_size, stride, padding, affine=True):
    """Create group of Convolution + BN + Relu function."""
    return ConvBnRelu(C_in, C_out, kernel_size, stride, padding, affine=affine)


@ClassFactory.register(ClassType.NETWORK)
class ConvBnRelu(ops.Module):
    """Create group of Convolution + BN + Relu."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, use_relu6=False):
        """Construct ConvBnRelu class."""
        super(ConvBnRelu, self).__init__()
        self.conv2d = ops.Conv2d(
            C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.batch_norm2d = ops.BatchNorm2d(C_out, affine=affine)
        if use_relu6:
            self.relu = ops.Relu6(inplace=False)
        else:
            self.relu = ops.Relu(inplace=False)


@ClassFactory.register(ClassType.NETWORK)
class SeparatedConv(ops.Module):
    """Separable convolution block with repeats."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1, affine=True, repeats=1):
        """Construct SepConv class."""
        super(SeparatedConv, self).__init__()
        for idx in range(repeats):
            self.add_module('{}_conv1'.format(idx),
                            ops.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                                       dilation=dilation, groups=C_in, bias=False))
            self.add_module('{}_conv2'.format(idx), ops.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False))
            self.add_module('{}_batch'.format(idx), ops.BatchNorm2d(C_in, affine=affine))
            self.add_module('{}_relu'.format(idx), ops.Relu(inplace=False))


@ClassFactory.register(ClassType.NETWORK)
class DilConv(ops.Module):
    """Separable convolution block with repeats."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        """Construct SepConv class."""
        super(DilConv, self).__init__()
        self.relu = ops.Relu(inplace=False)
        self.conv1 = ops.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=C_in, bias=False)
        self.conv2 = ops.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
        self.batch = ops.BatchNorm2d(C_out, affine=affine)


class GAPConv1x1(ops.Module):
    """Global Average Pooling + conv1x1."""

    def __init__(self, C_in, C_out):
        """Construct GAPConv1x1 class.

        :param C_in: input channel
        :param C_out: output channel
        """
        super(GAPConv1x1, self).__init__()
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, stride=1, padding=0)

    def call(self, x, *args, **kwargs):
        """Call GAPConv1x1."""
        size = ops.get_shape(x)[2:]
        out = x
        for model in self.children():
            out = ops.mean(out)
            out = model(out)
            out = ops.interpolate(out, size)
        return out


@ClassFactory.register(ClassType.NETWORK)
class FactorizedReduce(ops.Module):
    """Factorized reduce block."""

    def __init__(self, C_in, C_out, affine=True):
        """Construct FactorizedReduce class.

        :param C_in: input channel
        :param C_out: output channel
        :param affine: whether to use affine in BN
        """
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = ops.Relu(inplace=False)
        self.conv_1 = ops.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = ops.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = ops.BatchNorm2d(C_out, affine=affine)

    def call(self, x):
        """Do an inference on FactorizedReduce."""
        x = self.relu(x)
        out = ops.concat(tuple([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])]))
        out = self.bn(out)
        return out


@ClassFactory.register(ClassType.NETWORK)
class ReLUConvBN(ops.Module):
    """Class of ReLU + Conv + BN."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        """Init ReLUConvBN."""
        super(ReLUConvBN, self).__init__()
        self.relu = ops.Relu(inplace=False)
        self.conv = ops.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = ops.BatchNorm2d(C_out, affine=affine)


@ClassFactory.register(ClassType.NETWORK)
class Seq(ops.Module):
    """Separable convolution block with repeats."""

    def __init__(self, *models):
        """Construct SepConv class."""
        super(Seq, self).__init__()
        for idx, model in enumerate(models):
            self.add_module(str(idx), model)
