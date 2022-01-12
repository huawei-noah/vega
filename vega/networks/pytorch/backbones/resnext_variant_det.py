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

"""ResNeXtVariant for Detection."""
import math
import torch.nn as nn
from vega.common import ClassType, ClassFactory
from .resnet_variant_det import Bottleneck as _Bottleneck
from .resnet_variant_det import BasicBlock as _BasicBlock
from ..blocks.layer_creator import LayerCreator
from .resnet_variant_det import ResNetVariantDet


class BasicBlock(_BasicBlock):
    """Class of BasicBlock block for ResNeXt.

    :param inplanes: input feature map channel num
    :type inplanes: int

    :param planes: output feature map channel num
    :type planes: int

    :param groups: group num
    :type groups: int

    :param base_width: base width each group
    :type base_width: int

    :param base_channel: base channel of the model
    :type base_channel: int

    :param style: style,
    "pytorch" mean the stride-two layer is the 3x3 conv layer and
    "caffe" mean the stride-two layer is the first 1x1 conv layer
    :type style: str
    """

    def __init__(self, inplanes, planes, groups=1, base_width=4, base_channel=64, **kwargs):
        super(BasicBlock, self).__init__(inplanes, planes, **kwargs)
        self.__dict__.update(kwargs)
        self.planes = planes
        self.inplanes = inplanes
        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes * (base_width / base_channel)) * groups
        norm_creator = LayerCreator(**self.norm_cfg)
        conv_creator = LayerCreator(**self.conv_cfg)
        self.norm1_name = norm_creator.get_name(magic_number=1)
        norm1 = norm_creator.create_layer(num_features=width)
        self.norm2_name = norm_creator.get_name(magic_number=2)
        norm2 = norm_creator.create_layer(num_features=self.planes * self.expansion)
        self.conv1 = conv_creator.create_layer(self.inplanes, width, 3, stride=self.stride,
                                               padding=self.dilation, dilation=self.dilation, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv_creator.create_layer(width, self.planes * self.expansion, 3, padding=self.dilation,
                                               dilation=self.dilation, groups=groups, bias=False)
        self.add_module(self.norm2_name, norm2)


class Bottleneck(_Bottleneck):
    """Class of Bottleneck block for ResNeXt.

    :param inplanes: input feature map channel num
    :type inplanes: int

    :param planes: output feature map channel num
    :type planes: int

    :param groups: group num
    :type groups: int

    :param base_width: base width each group
    :type base_width: int

    :param base_channel: base channel of the model
    :type base_channel: int

    :param style: style,
    "pytorch" mean the stride-two layer is the 3x3 conv layer and
    "caffe" mean the stride-two layer is the first 1x1 conv layer
    :type style: str
    """

    def __init__(self, inplanes, planes, groups=2, base_width=32, base_channel=64, **kwargs):
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes * (base_width / base_channel)) * groups
        norm_creator = LayerCreator(**self.norm_cfg)
        conv_creator = LayerCreator(**self.conv_cfg)
        self.norm1_name = norm_creator.get_name(magic_number=1)
        norm1 = norm_creator.create_layer(num_features=width)
        self.norm2_name = norm_creator.get_name(magic_number=2)
        norm2 = norm_creator.create_layer(num_features=width)
        self.norm3_name = norm_creator.get_name(magic_number=3)
        norm3 = norm_creator.create_layer(num_features=self.planes * self.expansion)
        self.conv1 = conv_creator.create_layer(self.inplanes, width, kernel_size=1,
                                               stride=self.conv1_stride, bias=False)
        self.add_module(self.norm1_name, norm1)

        self.conv2 = conv_creator.create_layer(width, width, kernel_size=3, stride=self.conv2_stride,
                                               padding=self.dilation, dilation=self.dilation,
                                               groups=groups, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = conv_creator.create_layer(width, self.planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)


def make_res_layer(block,
                   inplanes,
                   planes,
                   arch,
                   groups=1,
                   base_width=4,
                   base_channel=64,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   conv_cfg=None,
                   norm_cfg=None,
                   ):
    """Make res layer.

    :param block: block function
    :type block: nn.Module
    :param inplanes: input feature map channel num
    :type inplanes: int
    :param planes: output feature map channel num
    :type planes: int
    :param arch: model arch
    :type arch: list
    :param groups: group num
    :type groups: int
    :param base_width: base width
    :type base_width: int
    :param base_channel: base channel
    :type base_channel: int
    :param stride: stride
    :type stride: int
    :param dilation: dilation
    :type dilation: int
    :param style: style
    :type style: str
    :param conv_cfg: conv config
    :type conv_cfg: dict
    :param norm_cfg: norm config
    :type norm_cfg: dict

    :return: res layer
    :rtype: nn.Module
    """
    conv_creator = LayerCreator(**conv_cfg)
    norm_creator = LayerCreator(**norm_cfg)
    layers = []
    for i, layer_type in enumerate(arch):
        downsample = None
        stride = stride if i == 0 else 1
        if layer_type == 2:
            planes *= 2
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_creator.create_layer(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                norm_creator.create_layer(num_features=planes * block.expansion))
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                groups=groups,
                base_width=base_width,
                base_channel=base_channel,
                downsample=downsample,
                style=style,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            ))
        inplanes = planes * block.expansion
    return nn.Sequential(*layers)


@ClassFactory.register(ClassType.NETWORK)
class ResNeXtVariantDet(ResNetVariantDet):
    """ResNeXtVariantDet backbone.

    :param net_desc: Description of ResNeXtVariantDet.
    :type net_desc: NetworkDesc
    """

    arch_settings = {18: BasicBlock,
                     34: BasicBlock,
                     50: Bottleneck,
                     101: Bottleneck,
                     152: Bottleneck}

    def __init__(self, desc):
        super(ResNeXtVariantDet, self).__init__(desc)
        self.base_channel = self.groups * self.base_width
        self.res_layers = []
        self.block = self.arch_settings[self.base_depth]
        total_expand = 0
        inplanes = planes = self.base_channel
        for i, arch in enumerate(self.arch):
            num_expand = arch.count(2)
            total_expand += num_expand
            stride = self.strides[i]
            dilation = self.dilations[i]
            res_layer = make_res_layer(
                self.block,
                inplanes,
                planes,
                arch,
                groups=self.groups,
                base_width=self.base_width,
                base_channel=self.base_channel,
                stride=stride,
                dilation=dilation,
                style=self.style,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
            )
            planes = self.base_channel * 2 ** total_expand
            inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self._freeze_stages()
        self.feat_dim = self.block.expansion * self.base_channel * 2 ** total_expand
