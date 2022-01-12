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

"""This is SearchSpace for blocks."""
from vega.common import ClassType, ClassFactory
from vega.modules.module import Module
from vega.modules.connections import Add
from vega.modules.operators import ops


@ClassFactory.register(ClassType.NETWORK)
class ShortCut(Module):
    """Create Shortcut SearchSpace."""

    def __init__(self, inchannel, outchannel, expansion, stride=1, norm_layer=None):
        """Create ShortCut layer.

        :param inchannel: input channel.
        :type inchannel: int
        :param outchannel: output channel.
        :type outchannel: int
        :param expansion: expansion
        :type expansion: int
        :param stride: the number to jump, default 1
        :type stride: int
        """
        super(ShortCut, self).__init__()
        if norm_layer is None:
            norm_layer = {"norm_type": 'BN'}
        if stride != 1 or inchannel != outchannel * expansion:
            self.conv1 = ops.Conv2d(in_channels=inchannel, out_channels=outchannel * expansion, kernel_size=1,
                                    stride=stride, bias=False)
            self.batch = build_norm_layer(features=outchannel * expansion, **norm_layer)
        else:
            self.identity = ops.Identity()


@ClassFactory.register(ClassType.NETWORK)
class BottleConv(Module):
    """Create BottleConv Searchspace."""

    def __init__(self, inchannel, outchannel, expansion, groups, base_width, stride=1, norm_layer=None,
                 Conv2d='Conv2d'):
        """Create BottleConv layer.

        :param inchannel: input channel.
        :type inchannel: int
        :param outchannel: output channel.
        :type outchannel: int
        :param expansion: expansion
        :type expansion: int
        :param stride: the number to jump, default 1
        :type stride: int
        """
        super(BottleConv, self).__init__()
        if norm_layer is None:
            norm_layer = {"norm_type": 'BN'}
        outchannel = int(outchannel * (base_width / 64.)) * groups
        self.conv1 = build_conv_layer(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=1,
                                      bias=False, Conv2d=Conv2d)
        self.batch1 = build_norm_layer(features=outchannel, **norm_layer)
        self.relu1 = ops.Relu(inplace=True)
        self.conv2 = build_conv_layer(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=stride,
                                      padding=1, groups=groups, bias=False, Conv2d=Conv2d)
        self.batch2 = build_norm_layer(features=outchannel, **norm_layer)
        self.relu2 = ops.Relu(inplace=True)
        self.conv3 = build_conv_layer(in_channels=outchannel, out_channels=outchannel * expansion, kernel_size=1,
                                      stride=1, bias=False, Conv2d=Conv2d)
        self.batch3 = build_norm_layer(features=outchannel * expansion, **norm_layer)


@ClassFactory.register(ClassType.NETWORK)
class BasicConv(Module):
    """Create BasicConv Searchspace."""

    def __init__(self, inchannel, outchannel, groups=1, base_width=64, stride=1, norm_layer=None,
                 Conv2d='Conv2d'):
        """Create BasicConv layer.

        :param inchannel: input channel.
        :type inchannel: int
        :param outchannel: output channel.
        :type outchannel: int
        :param stride: the number to jump, default 1
        :type stride: int
        """
        super(BasicConv, self).__init__()
        if norm_layer is None:
            norm_layer = {"norm_type": 'BN'}
        self.conv = build_conv_layer(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=stride,
                                     padding=1, groups=groups, bias=False, Conv2d=Conv2d)
        self.batch = build_norm_layer(features=outchannel, **norm_layer)
        self.relu = ops.Relu(inplace=True)
        self.conv2 = build_conv_layer(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1,
                                      padding=1, groups=groups, bias=False, Conv2d=Conv2d)
        self.batch2 = build_norm_layer(features=outchannel, **norm_layer)


@ClassFactory.register(ClassType.NETWORK)
class SmallInputInitialBlock(Module):
    """Create SmallInputInitialBlock SearchSpace."""

    def __init__(self, init_plane):
        """Create SmallInputInitialBlock layer.

        :param init_plane: input channel.
        :type init_plane: int
        """
        super(SmallInputInitialBlock, self).__init__()
        self.conv = ops.Conv2d(in_channels=3, out_channels=init_plane, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn = ops.BatchNorm2d(num_features=init_plane)
        self.relu = ops.Relu()


@ClassFactory.register(ClassType.NETWORK)
class InitialBlock(Module):
    """Create InitialBlock SearchSpace."""

    def __init__(self, init_plane):
        """Create InitialBlock layer.

        :param init_plane: input channel.
        :type init_plane: int
        """
        super(InitialBlock, self).__init__()
        self.conv = ops.Conv2d(in_channels=3, out_channels=init_plane, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.batch = ops.BatchNorm2d(num_features=init_plane)
        self.relu = ops.Relu()
        self.maxpool2d = ops.MaxPool2d(kernel_size=3, stride=2, padding=1)


@ClassFactory.register(ClassType.NETWORK)
class BasicBlock(Module):
    """Create BasicBlock SearchSpace."""

    expansion = 1

    def __init__(self, inchannel, outchannel, groups=1, base_width=64, stride=1, norm_layer=None,
                 Conv2d='Conv2d'):
        """Create BasicBlock layers.

        :param inchannel: input channel.
        :type inchannel: int
        :param outchannel: output channel.
        :type outchannel: int
        :param stride: the number to jump, default 1
        :type stride: int
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = {"norm_type": 'BN'}
        base_conv = BasicConv(inchannel=inchannel, outchannel=outchannel, stride=stride,
                              groups=groups, base_width=base_width, norm_layer=norm_layer, Conv2d=Conv2d)
        shortcut = ShortCut(inchannel=inchannel, outchannel=outchannel, expansion=self.expansion,
                            stride=stride, norm_layer=norm_layer)
        self.block = Add(base_conv, shortcut)
        self.relu = ops.Relu()


@ClassFactory.register(ClassType.NETWORK)
class BottleneckBlock(Module):
    """Create BottleneckBlock SearchSpace."""

    expansion = 4

    def __init__(self, inchannel, outchannel, groups=1, base_width=64, stride=1, norm_layer=None,
                 Conv2d='Conv2d'):
        """Create BottleneckBlock layers.

        :param inchannel: input channel.
        :type inchannel: int
        :param outchannel: output channel.
        :type outchannel: int
        :param stride: the number to jump, default 1
        :type stride: int
        """
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = {"norm_type": 'BN'}
        bottle_conv = BottleConv(inchannel=inchannel, outchannel=outchannel, expansion=self.expansion,
                                 stride=stride, groups=groups, base_width=base_width, norm_layer=norm_layer,
                                 Conv2d=Conv2d)
        shortcut = ShortCut(inchannel=inchannel, outchannel=outchannel, expansion=self.expansion, stride=stride,
                            norm_layer=norm_layer)
        self.block = Add(bottle_conv, shortcut)
        self.relu = ops.Relu()


@ClassFactory.register(ClassType.NETWORK)
class PruneBasicBlock(Module):
    """Basic block class in prune resnet."""

    expansion = 1

    def __init__(self, inchannel, outchannel, innerchannel, stride=1):
        """Init PruneBasicBlock."""
        super(PruneBasicBlock, self).__init__()
        conv_block = PruneBasicConv(inchannel, outchannel, innerchannel, stride)
        shortcut = ShortCut(inchannel, outchannel, self.expansion, stride)
        self.block = Add(conv_block, shortcut)
        self.relu3 = ops.Relu()


@ClassFactory.register(ClassType.NETWORK)
class PruneBasicConv(Module):
    """Create PruneBasicConv Searchspace."""

    def __init__(self, in_planes, planes, inner_plane, stride=1):
        """Create BottleConv layer."""
        super(PruneBasicConv, self).__init__()
        self.conv1 = ops.Conv2d(
            in_planes, inner_plane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = ops.BatchNorm2d(inner_plane)
        self.relu = ops.Relu()
        self.conv2 = ops.Conv2d(inner_plane, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = ops.BatchNorm2d(planes)
        self.relu2 = ops.Relu()


@ClassFactory.register(ClassType.NETWORK)
class TextConvBlock(Module):
    """Create Conv Block in text CNN."""

    def __init__(self, in_channels=1, out_channels=16, kernel_size=(3, 3)):
        super(TextConvBlock, self).__init__()
        self.conv1 = ops.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.squeeze = ops.Squeeze(3)
        self.relu = ops.Relu()
        self.max_pool = ops.GlobalMaxPool1d()
        self.squeeze2 = ops.Squeeze(-1)


def build_norm_layer(features, norm_type='BN', **kwargs):
    """Build norm layers according to their type.

    :param features: input tensor.
    :param norm_type: type of norm layer.
    :param **kwargs: other optional parameters.
    """
    if norm_type == 'BN':
        return ops.BatchNorm2d(features, **kwargs)
    elif norm_type == 'GN':
        if 'num_groups' in kwargs.keys():
            num_groups = kwargs.pop('num_groups')
            return ops.GroupNorm(num_groups, features, **kwargs)
        else:
            raise ValueError('Num_groups is required for group normalization')
    elif norm_type == 'Sync':
        return ops.SyncBatchNorm(features, **kwargs)
    else:
        raise ValueError('norm type {} is not defined'.format(norm_type))


def build_conv_layer(in_channels, out_channels, kernel_size, bias, Conv2d, padding=0, groups=1, stride=1):
    """Build conv layers according to their type.

    :param features: input tensor.
    :param norm_type: type of norm layer.
    :param **kwargs: other optional parameters.
    """
    if Conv2d == 'Conv2d':
        return ops.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, groups=groups, bias=bias)
    elif Conv2d == 'ConvWS2d':
        return ops.ConvWS2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=groups, bias=bias)
