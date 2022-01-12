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

"""CurveLane neck for detection."""
import torch
import torch.nn as nn
from vega.common import ClassType, ClassFactory
from ..blocks.layer_creator import LayerCreator


class ConvPack(nn.Module):
    """ConvPack.

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

    :return: Conv pack layer
    :rtype: nn.Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu',
                 inplace=True):
        super().__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        conv_creator = LayerCreator(**conv_cfg)
        self.conv = conv_creator.create_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.with_norm:
            norm_channels = out_channels
            norm_creator = LayerCreator(**norm_cfg)
            norm = norm_creator.create_layer(num_features=norm_channels)
            self.norm_name = norm_creator.get_name()
            self.add_module(self.norm_name, norm)
        if self.with_activatation:
            act_cfg = {'type': 'ReLU'}
            act_creator = LayerCreator(**act_cfg)
            self.activate = act_creator.create_layer(inplace=inplace)

    def norm(self, x):
        """Apply norm."""
        x = getattr(self, self.norm_name)(x)
        return x

    def forward(self, x, activate=True, norm=True):
        """Forward compute.

        :param x: input feature map
        :type x: tensor
        :param activate: whether activate or not
        :type activate: bool
        :param norm: whether norm or not
        :type norm: bool
        :return: output feature map
        :rtype: tensor
        """
        x = self.conv(x)
        if norm and self.with_norm:
            x = self.norm(x)
        if activate and self.with_activatation:
            x = self.activate(x)
        return x


class FeatureFusionNetwork(nn.Module):
    """The Core of FeatureFusionNetwork.

    :param out_channels: out_channels
    :type out_channels: int
    :param num_outs: num_outs
    :type num_outs: int
    :param start_level: start_level
    :type start_level: int
    :param end_level: end_level
    :type end_level: int
    :param in_channels: in_channels
    :type in_channels: int
    :param add_extra_convs: add_extra_convs
    :type add_extra_convs: bool
    :param extra_convs_on_inputs: extra_convs_on_inputs
    :type extra_convs_on_inputs: bool
    :param relu_before_extra_convs: relu_before_extra_convs
    :type relu_before_extra_convs: bool
    :param conv_cfg: conv_cfg
    :type conv_cfg: dict
    :param norm_cfg: norm_cfg
    :type norm_cfg: dict
    :param activation: activation
    :type activation: dict
    :param feature_fusion_arch_str: feature_fusion_arch_str
    :type feature_fusion_arch_str: atr
    """

    def __init__(self,
                 out_channels=128,
                 num_outs=4,
                 start_level=0,
                 end_level=-1,
                 in_channels=None,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 feature_fusion_arch_str=None):
        super(FeatureFusionNetwork, self).__init__()
        if conv_cfg is None:
            conv_cfg = {'type': 'Conv'}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.feature_fusion_arch_str = feature_fusion_arch_str
        self.c34_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.c24_maxpool = nn.MaxPool2d(kernel_size=5, stride=4, padding=1)
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvPack(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvPack(
                out_channels * 2,
                out_channels * 2,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvPack(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def decoder_ffm_arch(self):
        """Decode ffm arch."""
        feature_fusion_arch = []
        block_arch = []
        for i in self.feature_fusion_arch_str:
            if i == '-':
                feature_fusion_arch.append(block_arch)
                block_arch = []
            else:
                block_arch.append(int(i))
        feature_fusion_arch.append(block_arch)
        return feature_fusion_arch

    def forward(self, inputs):
        """Forward method."""
        build_out = []
        fpn_arch = self.decoder_ffm_arch()
        for i in range(len(fpn_arch)):
            input1, input2 = fpn_arch[i][0], fpn_arch[i][1]
            laterals = [self.lateral_convs[input1](inputs[input1]), self.lateral_convs[input2](inputs[input2])]

            # sum of the two input
            if input1 == 0:
                laterals[0] = self.c24_maxpool(laterals[0])
            elif input1 == 1:
                laterals[0] = self.c34_maxpool(laterals[0])
            if input2 == 0:
                laterals[1] = self.c24_maxpool(laterals[1])
            elif input2 == 1:
                laterals[1] = self.c34_maxpool(laterals[1])

            build_out.append(self.fpn_convs[i](torch.cat((laterals[0], laterals[1]), 1)))

        outs = torch.cat((inputs[2], torch.cat((build_out[0], build_out[1]), 1)), 1)
        return outs


def PseudoFeatureFusionNetwork(feature_map_list):
    """Pseudo FeatureFusionNetwork, just get the third layer of target featuremap."""
    return feature_map_list[2]


def ArchChannels2Module(feature_fusion_arch_code, in_channels):
    """Ffn warpper."""
    if feature_fusion_arch_code != '-':
        return FeatureFusionNetwork(in_channels=in_channels,
                                    out_channels=64,
                                    num_outs=4,
                                    feature_fusion_arch_str=feature_fusion_arch_code)
    else:
        return PseudoFeatureFusionNetwork


@ClassFactory.register(ClassType.NETWORK)
class FeatureFusionModule(nn.Module):
    """FeatureFusionModule backbone.

    :param desc: Description of ResNetVariantDet.
    :type desc: NetworkDesc
    """

    def __init__(self, desc):
        super(FeatureFusionModule, self).__init__()
        self.in_channels = desc["in_channels"][0:4]
        self.feature_fusion_arch_code = desc["arch_code"]
        self.num_ins = len(self.in_channels)
        self.neck = ArchChannels2Module(self.feature_fusion_arch_code, self.in_channels)

    def forward(self, inputs):
        """Get the result of ffm."""
        out = self.neck(inputs[0:4])
        return out

    def init_weights(self):
        """Initialize ffm weight."""
        if self.feature_fusion_arch_code != '-':
            self.neck.init_weights()
