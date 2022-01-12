# -*- coding: utf-8 -*-

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

"""Import all torch operators."""
import math
from vega.common import ClassType, ClassFactory
from vega.modules.operators import ops


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
def conv_bn_relu6(C_in, C_out, kernel_size=3, stride=1, padding=0, affine=True,
                  groups=1, depthwise=False, inplace=False):
    """Create group of Convolution + BN + Relu6 function."""
    return ConvBnRelu(C_in, C_out, kernel_size, stride, padding, affine=affine, use_relu6=True,
                      groups=groups, depthwise=depthwise, inplace=False)


@ClassFactory.register(ClassType.NETWORK)
def conv_bn_relu(C_in, C_out, kernel_size, stride, padding, affine=True, groups=1, depthwise=False, inplace=False):
    """Create group of Convolution + BN + Relu function."""
    return ConvBnRelu(C_in, C_out, kernel_size, stride, padding, affine=affine,
                      groups=groups, depthwise=depthwise, inplace=False)


@ClassFactory.register(ClassType.NETWORK)
class ConvBnRelu(ops.Module):
    """Create group of Convolution + BN + Relu."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding, groups=1, depthwise=False,
                 Conv2d='Conv2d', affine=True, use_relu6=False, inplace=False, norm_layer='BN',
                 has_bn=True, has_relu=True, **kwargs):
        """Construct ConvBnRelu class."""
        super(ConvBnRelu, self).__init__()
        features = []
        conv2d = None
        if Conv2d == 'Conv2d':
            conv2d = ops.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False,
                                groups=groups, depthwise=depthwise)
        elif Conv2d == 'ConvWS2d':
            conv2d = ops.ConvWS2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False,
                                  groups=groups, depthwise=depthwise)
        if conv2d:
            features.append(conv2d)
        if has_bn:
            batch_norm2d = None
            if norm_layer == 'BN':
                batch_norm2d = ops.BatchNorm2d(C_out, affine=affine)
            elif norm_layer == 'GN':
                num_groups = kwargs.pop('num_groups')
                batch_norm2d = ops.GroupNorm(num_groups, C_out, affine=affine)
            elif norm_layer == 'Sync':
                batch_norm2d = ops.SyncBatchNorm(C_out, affine=affine)
            if batch_norm2d:
                features.append(batch_norm2d)
        if has_relu:
            if use_relu6:
                relu = ops.Relu6(inplace=inplace)
            else:
                relu = ops.Relu(inplace=inplace)
            features.append(relu)
        for idx, model in enumerate(features):
            self.add_module(str(idx), model)


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

    def call(self, x=None, *args, **kwargs):
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
        if C_out % 2 != 0:
            raise ValueError('Outchannel must be divided by 2.')
        self.relu = ops.Relu(inplace=False)
        self.conv_1 = ops.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = ops.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = ops.BatchNorm2d(C_out, affine=affine)

    def call(self, x=None, *args, **kwargs):
        """Do an inference on FactorizedReduce."""
        x = self.relu(x)
        out = ops.concat((self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])))
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


@ClassFactory.register(ClassType.NETWORK)
class GhostConv2d(ops.Module):
    """Ghost Conv2d Module."""

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, affine=True, padding=0, ratio=2):
        super(GhostConv2d, self).__init__()
        self.C_out = C_out
        init_channels = math.ceil(C_out / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = Seq(
            ops.Relu(inplace=False),
            ops.Conv2d(C_in, init_channels, kernel_size=1, stride=stride, padding=padding, bias=False),
            ops.BatchNorm2d(init_channels, affine=affine)
        )

        self.cheap_operation = Seq(
            ops.Relu(inplace=False),
            ops.Conv2d(init_channels, new_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                       groups=init_channels, bias=False),
            ops.BatchNorm2d(new_channels, affine=affine)
        )

    def call(self, x=None, *args, **kwargs):
        """Call function."""
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = ops.concat([x1, x2], dim=1)
        return out[:, :self.C_out, :, :]
