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

"""Decode and build BiSeNet."""

import torch.nn as nn
from vega.modules.operators import ConvBnRelu
from vega.modules.operators import conv3x3
from vega.modules.blocks import BasicBlock, BottleneckBlock, build_norm_layer
from .common import load_model


class ResNet_arch(nn.Module):
    """ResNet_arch module."""

    def __init__(self, block, arch, base_channel, strides=None,
                 dilations=None, num_classes=1000, groups=1, base_width=64,
                 structure='full', Conv2d='Conv2d', norm_layer=None):
        """Construct the ResNet_arch class.

        :param block: BasicBlock or Bottleneck instance
        :param arch: code of model
        :param base_channel: base channel numbers
        :param stride: stride of the convolution
        :param dilation: dilation of convolution layer
        :param num_classes: number of output classes
        :param groups: groups of convolution layer
        :param base_width: base channel numbers
        :param structure: structure of the model
        :param norm_layer: type of norm layer.
        :param Conv2d: type of conv layer.
        """
        if strides is None:
            strides = [1, 2, 2, 2]
        if dilations is None:
            dilations = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = {"norm_type": 'BN'}
        if structure in ['full', 'drop_last', 'backbone']:
            self.structure = structure
            self.num_classes = num_classes
            self.arch = [[int(a) for a in x] for x in arch.split('-')]
            self.base_channel = base_channel
            self.strides = strides
            self.dilations = dilations
            super(ResNet_arch, self).__init__()
            self.conv1 = conv3x3(3, base_channel // 2, stride=2)
            self.bn1 = build_norm_layer((base_channel // 2), **norm_layer)
            self.relu = nn.ReLU(inplace=False)
            self.conv2 = conv3x3(base_channel // 2, base_channel, stride=2)
            self.bn2 = build_norm_layer((base_channel), **norm_layer)
            self.res_layers = []
            self.block = block
            total_expand = 0
            inplanes = planes = self.base_channel
            self.stage_out_channels = []
            for i, arch in enumerate(self.arch):
                num_expand = arch.count(2)
                total_expand += num_expand
                stride = self.strides[i]
                res_layer, out_channels = self.make_res_layer(
                    self.block,
                    inplanes,
                    planes,
                    arch,
                    groups=groups,
                    base_width=base_width,
                    stride=stride,
                    norm_layer=norm_layer,
                    Conv2d=Conv2d)
                self.stage_out_channels.append(out_channels)
                planes = self.base_channel * 2 ** total_expand
                inplanes = planes * self.block.expansion
                layer_name = 'layer{}'.format(i + 1)
                self.add_module(layer_name, res_layer)
                self.res_layers.append(layer_name)
            self.out_channels = out_channels
        else:
            raise ValueError('unknown structrue: %s' % repr(structure))

    def get_output_size(self, H=None, W=None):
        """Get size of the output.

        :param H: input height
        :param W: input width
        :return: size of the output
        """
        if self.structure == 'full':
            return (self.num_classes,)
        elif self.structure == 'drop_last':
            return (self.out_channels,)
        else:
            if None not in [H, W]:
                return (self.out_channels, H // 32, W // 32)
            else:
                raise ValueError('requires arguments H, W')

    @staticmethod
    def make_res_layer(block,
                       inplanes,
                       planes,
                       arch,
                       Conv2d,
                       groups=1,
                       base_width=64,
                       stride=1,
                       norm_layer='BN'):
        """Make resnet layers.

        :param block: BasicBlock or Bottleneck instance
        :param in_planes: input channels
        :param planes: output channels
        :param arch: code of model
        :param base_channel: base channel numbers
        :param groups: groups of convolution layer
        :param base_width: base channel numbers
        :param stride: stride of the convolution
        :param dilation: dilation of convolution layer
        :param Conv2d: type of conv layer.
        :param norm_layer: type of norm layer.
        """
        layers = []
        for i, layer_type in enumerate(arch):
            stride = stride if i == 0 else 1
            if layer_type == 2:
                planes *= 2
            layers.append(
                block(
                    inchannel=inplanes,
                    outchannel=planes,
                    stride=stride,
                    groups=groups,
                    base_width=base_width,
                    norm_layer=norm_layer,
                    Conv2d=Conv2d))
            inplanes = planes * block.expansion
        return nn.Sequential(*layers), inplanes

    def forward(self, x):
        """Get different stages' output.

        :param x: input tensor
        :return: different stages' output
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)
        return blocks

    @classmethod
    def set_from_arch_string(cls, arch_string, **kwargs):
        """From code string create model.

        :param arch_string: code strings to be decoded
        :return: model
        """
        params = kwargs
        params.update(arch_string)
        base_depth = params.pop('base_depth')
        if isinstance(base_depth, int):
            block = BasicBlock if base_depth <= 34 else BottleneckBlock
        else:
            block = BasicBlock if base_depth == "bb" else BottleneckBlock

        params.update(block=block)
        return cls(**params)


def build_archs(arch_string, pretrained_model=None, num_classes=1000, structure='full',
                Conv2d='Conv2d', norm_layer='BN', **kwargs):
    """From code string create model and load model.

    :param arch_string: code strings to be decoded
    :param pretrained_model: whether use pretrained model
    :param num_classes: number of output classes
    :param structure: structure of the model
    :param Conv2d: type of conv layer.
    :param norm_layer: type of norm layer.
    :param **kwargs: other keywords.
    """
    model = ResNet_arch.set_from_arch_string(arch_string, num_classes=num_classes, structure=structure,
                                             Conv2d=Conv2d, norm_layer=norm_layer)
    if pretrained_model == 'None':
        pretrained_model = None
    if pretrained_model:
        model = load_model(model, pretrained_model)
    return model


class AutoSpatialPath(nn.Module):
    """Build spatial path from code string."""

    def __init__(self, layer, arch, norm_layer='BN', Conv2d=nn.Conv2d, stride=None, **kwargs):
        """Build spatial path.

        :param layer: layers of spatial path
        :param arch: code of model
        :param norm_layer: type of norm layer.
        :param Conv2d: type of conv layer.
        :param stride: stride of the convolution
        :param **kwargs: other keywords.
        :return: output tensor
        """
        if stride is None:
            stride = [1, 2, 2, 1]
        super(AutoSpatialPath, self).__init__()
        split_arch = arch.split('_')
        self.base_channels = int(split_arch[0])

        arch = [[int(a) for a in x] for x in split_arch[1].split('-')]
        self.conv1 = ConvBnRelu(3, self.base_channels, 3, 2, 1, norm_layer=norm_layer, **kwargs)
        self.arch = arch
        self.stride = stride
        self.layer = layer
        self.layers = []
        in_channels = self.base_channels
        for i, arch in enumerate(self.arch):
            layer, in_channels = self.make_spatial_layer(
                self.layer,
                arch,
                in_channels,
                stride=self.stride[i],
                norm_layer=norm_layer,
                Conv2d=Conv2d)
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)

    def make_spatial_layer(self, layer, arch, in_channels, stride=1, Conv2d=nn.Conv2d, norm_layer='BN'):
        """Make spatial layers.

        :param layer: layers of spatial path
        :param arch: code of spatial path
        :param in_channels: input channels
        :param stride: stride of the convolution
        :param Conv2d: type of conv layer.
        :param norm_layer: type of norm layer.
        :return: output tensor
        """
        layers = []
        out_channels = in_channels * arch[0]
        layers.append(layer(in_channels, out_channels, 3, stride, 1,
                            has_bn=True, norm_layer=norm_layer,
                            has_relu=True, has_bias=False, Conv2d=Conv2d))
        in_channels = out_channels
        for i, layer_type in enumerate(arch[1:]):
            out_channels = in_channels * layer_type
            layers.append(layer(in_channels, out_channels, 3, 1, 1,
                                has_bn=True, norm_layer=norm_layer,
                                has_relu=True, has_bias=False, Conv2d=Conv2d))
            in_channels = out_channels
        return nn.Sequential(*layers), out_channels

    def forward(self, x):
        """Do an inference on spatial layers.

        :param x: input tensor
        :return: output tensor
        """
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def build_spatial_path(string, Conv2d=nn.Conv2d, norm_layer='BN', **kwargs):
    """Call the function to build spatial layers."""
    return AutoSpatialPath(ConvBnRelu, string, norm_layer=norm_layer, Conv2d=Conv2d, **kwargs)
