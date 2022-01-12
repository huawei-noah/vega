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

"""The BiSeNet model."""
import torch.nn as nn
import torch.nn.functional as F
from vega.common import ClassType, ClassFactory
from vega.modules.module import Module
from vega.modules.operators import ConvBnRelu
from .segmentation.common import AttentionRefinement, FeatureFusion
from .segmentation.evolveresnet import build_archs, build_spatial_path
from .segmentation.weights import init_weight


@ClassFactory.register(ClassType.NETWORK)
class BiSeNet(Module):
    """BiSeNet module."""

    def __init__(self, **desc):
        """Construct the BiSeNet class.

        :param desc: configs of BiSeNet from .yml file
        """
        self.out_planes = desc['num_class']
        self.conv = desc['conv']
        self.conv_channel = desc['conv_channel']
        self.norm_layer = desc['norm_layer']
        self.backone_args = desc['backbone_args']
        self.encoding = desc['config']
        norm_type = self.norm_layer['norm_type']
        if norm_type == 'GN':
            self.norm_op = nn.GroupNorm
        elif norm_type == 'BN':
            self.norm_op = nn.BatchNorm2d
        elif norm_type == 'Sync':
            self.norm_op = nn.SyncBatchNorm
        else:
            raise NotImplementedError
        super(BiSeNet, self).__init__()
        self.decode_model(self.encoding)

    def decode_model(self, encoding):
        """Creatr 'BiSeNet' from encoding.

        :param encoding: encoding of 'BiSeNet'
        """
        context_path = encoding[0]
        spatial_path = encoding[1]
        if self.conv == 'Conv2d':
            Conv2d = 'Conv2d'
        elif self.conv == 'ConvWS2d':
            Conv2d = 'ConvWS2d'
        else:
            raise ValueError('Convolution layer {} is not defined'.format(self.conv))

        self.context_path = build_archs(arch_string=context_path, Conv2d=Conv2d, norm_layer=self.norm_layer,
                                        **self.backone_args)

        if spatial_path is None:
            self.spatial_path = SpatialPath(3, self.conv_channel, norm_layer=self.norm_layer,
                                            Conv2d=Conv2d)
        else:
            self.spatial_path = build_spatial_path(spatial_path, norm_layer=self.norm_layer,
                                                   Conv2d=Conv2d)
        self.business_layer = []
        conv_channel = self.conv_channel
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(self.context_path.out_channels, conv_channel, 1, 1, 0,
                       norm_layer=self.norm_layer, Conv2d=Conv2d)
        )
        arms = [AttentionRefinement(self.context_path.stage_out_channels[-1], conv_channel,
                                    self.norm_layer, Conv2d=Conv2d),
                AttentionRefinement(self.context_path.stage_out_channels[-2], conv_channel,
                                    self.norm_layer, Conv2d=Conv2d)]
        refines = [ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              norm_layer=self.norm_layer,
                              Conv2d=Conv2d),
                   ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              norm_layer=self.norm_layer,
                              Conv2d=Conv2d)]

        heads = [BiSeNetHead(conv_channel, self.out_planes, 16,
                             True, self.norm_layer, Conv2d=Conv2d),
                 BiSeNetHead(conv_channel, self.out_planes, 8,
                             True, self.norm_layer, Conv2d=Conv2d),
                 BiSeNetHead(conv_channel * 2, self.out_planes, 8,
                             False, self.norm_layer, Conv2d=Conv2d)]
        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2,
                                 4, self.norm_layer)
        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)
        self.business_layer.append(self.spatial_path)
        self.business_layer.append(self.global_context)
        self.business_layer.append(self.arms)
        self.business_layer.append(self.refines)
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)

    def init_weight(self):
        """Init norm layer."""
        momentum = self.norm_layer.get('momentum')
        init_weight(self.business_layer, nn.init.kaiming_normal_,
                    self.norm_op, self.norm_layer['eps'],
                    momentum,
                    mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        """Do an inference on BiSeNet.

        :param x: input tensor
        :return: list of output tensor
        """
        data = x
        spatial_out = self.spatial_path(data)
        context_blocks = self.context_path(data)
        context_blocks.reverse()
        global_context = self.global_context(context_blocks[0])
        global_context = F.interpolate(global_context,
                                       size=context_blocks[0].size()[2:],
                                       mode='bilinear', align_corners=True)
        last_fm = global_context
        pred_out = []
        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms,
                                                  self.refines)):
            fm = arm(fm)
            fm += last_fm
            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]),
                                    mode='bilinear', align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm
        concate_fm = self.ffm(spatial_out, context_out)
        pred_out.append(concate_fm)
        return [self.heads[0](pred_out[0]), self.heads[1](pred_out[1]), self.heads[-1](pred_out[2])]


class SpatialPath(nn.Module):
    """SpatialPath module."""

    def __init__(self, in_planes, out_planes, norm_layer='BN', Conv2d=nn.Conv2d, inner_channel=64, **kwargs):
        """Create SpatialPath.

        :param in_planes: input channels
        :param out_planes: output channels
        :param norm_layer: type of norm layer.
        :param Conv2d: type of conv layer.
        :param inner_channel: number of inner channels.
        """
        super(SpatialPath, self).__init__()
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                   norm_layer=norm_layer,
                                   Conv2d=Conv2d)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     norm_layer=norm_layer,
                                     Conv2d=Conv2d)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     norm_layer=norm_layer,
                                     Conv2d=Conv2d)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                                   norm_layer=norm_layer,
                                   Conv2d=Conv2d)

    def forward(self, x):
        """Do an inference on SpatialPath.

        :param x: input tensor
        :return: output tensor
        """
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        x = self.conv_1x1(x)

        return x


class BiSeNetHead(nn.Module):
    """BiSeNetHead module."""

    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer='BN', Conv2d=nn.Conv2d):
        """Create BiSeNetHead.

        :param in_planes: input channels
        :param out_planes: output channels
        :param scale: scale factor.
        :param is_aux: whether use aux weight.
        :param norm_layer: type of norm layer.
        :param Conv2d: type of conv layer.
        """
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 256, 3, 1, 1,
                                       norm_layer=norm_layer,
                                       Conv2d=Conv2d)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       norm_layer=norm_layer,
                                       Conv2d=Conv2d)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(256, out_planes, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1, stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        """Do an inference on SpatialPath.

        :param x: input tensor
        :return: output tensor
        """
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)
        return output
