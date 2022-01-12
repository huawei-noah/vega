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

"""ResNetVariant for Detection."""
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from vega.common import ClassType, ClassFactory
from vega.networks.pytorch.blocks.layer_creator import LayerCreator


class BasicBlock(nn.Module):
    """Class of BasicBlock block.

    :param inplanes: input feature map channel num
    :type inplanes: int

    :param planes: output feature map channel num
    :type planes: int

    :param stride: stride
    :type stride: int

    :param dilation: dilation
    :type dilation: int

    :param downsample: downsample
    :type downsample: a block to execute downsample operate.

    :param style: style,
    "pytorch" mean the stride-two layer is the 3x3 conv layer and
    "caffe" mean the stride-two layer is the first 1x1 conv layer.
    :type style: str

    :param conv_cfg: conv config
    :type conv_cfg: dict

    :param norm_cfg: norm config
    :type norm_cfg: dict
    """

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=None, **kwargs):
        super(BasicBlock, self).__init__()
        norm_creator = LayerCreator(**norm_cfg)
        self.norm1_name = norm_creator.get_name(magic_number=1)
        norm1 = norm_creator.create_layer(num_features=planes)
        self.add_module(self.norm1_name, norm1)
        self.norm2_name = norm_creator.get_name(magic_number=2)
        norm2 = norm_creator.create_layer(num_features=planes)

        conv_creator = LayerCreator(**conv_cfg)
        self.conv1 = conv_creator.create_layer(inplanes, planes, 3, stride=stride, padding=dilation,
                                               dilation=dilation, bias=False)
        self.conv2 = conv_creator.create_layer(planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def _get_module_by_name(self, module_name):
        return self.__getattr__(module_name)

    def norm1(self, x):
        """Apply norm1."""
        return self._get_module_by_name(self.norm1_name)(x)

    def norm2(self, x):
        """Apply norm2."""
        return self._get_module_by_name(self.norm2_name)(x)

    def forward(self, x):
        """Forward compute.

        :param x: input feature map
        :return: output feature map
        :rtype: tensor
        """
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Class of Bottleneck block.

    :param inplanes: input feature map channel num
    :type inplanes: int

    :param planes: output feature map channel num
    :type planes: int

    :param stride: stride
    :type stride: int

    :param dilation: dilation
    :type dilation: int

    :param base_width: base width each group
    :type base_width: int

    :param style: style,
    "pytorch" mean the stride-two layer is the 3x3 conv layer and
    "caffe" mean the stride-two layer is the first 1x1 conv layer
    :type style: str

    :param conv_cfg: conv config
    :type conv_cfg: dict

    :param norm_cfg: norm config
    :type norm_cfg: dict
    """

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=None, **kwargs):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        norm_creator = LayerCreator(**norm_cfg)
        self.norm1_name = norm_creator.get_name(magic_number=1)
        norm1 = norm_creator.create_layer(num_features=planes)
        self.norm2_name = norm_creator.get_name(magic_number=2)
        norm2 = norm_creator.create_layer(num_features=planes)
        self.norm3_name = norm_creator.get_name(magic_number=3)
        norm3 = norm_creator.create_layer(num_features=planes * self.expansion)
        conv_creator = LayerCreator(**conv_cfg)
        self.conv1 = conv_creator.create_layer(inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv_creator.create_layer(planes, planes, kernel_size=3, stride=self.conv2_stride,
                                               padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = conv_creator.create_layer(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def _get_module_by_name(self, module_name):
        return self.__getattr__(module_name)

    def norm1(self, x):
        """Apply norm1."""
        return self._get_module_by_name(self.norm1_name)(x)

    def norm2(self, x):
        """Apply norm2."""
        return self._get_module_by_name(self.norm2_name)(x)

    def norm3(self, x):
        """Apply norm3."""
        return self._get_module_by_name(self.norm3_name)(x)

    def forward(self, x):
        """Forward compute.

        :param x: input feature map
        :return: output feature map
        :rtype: tensor
        """
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   arch,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   conv_cfg=None,
                   norm_cfg=None, ):
    """Make res layer.

    :param block: block function
    :type block: nn.Module
    :param inplanes: input feature map channel num
    :type inplanes: int
    :param planes: output feature map channel num
    :type planes: int
    :param arch: model arch
    :type arch: list
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
                conv_creator.create_layer(in_channels=inplanes, out_channels=planes * block.expansion, kernel_size=1,
                                          stride=stride, bias=False),
                norm_creator.create_layer(num_features=planes * block.expansion))
        layers.append(block(inplanes=inplanes, planes=planes, stride=stride, dilation=dilation, downsample=downsample,
                            style=style, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
        inplanes = planes * block.expansion
    return nn.Sequential(*layers)


@ClassFactory.register(ClassType.NETWORK)
class ResNetVariantDet(nn.Module):
    """ResNetVariantDet backbone.

    :param net_desc: Description of ResNetVariantDet.
    :type net_desc: NetworkDesc
    """

    arch_settings = {18: BasicBlock,
                     34: BasicBlock,
                     50: Bottleneck,
                     101: Bottleneck,
                     152: Bottleneck}

    def __init__(self, desc):
        super(ResNetVariantDet, self).__init__()

        self.__dict__.update(desc)
        if self.base_depth not in self.arch_settings:
            raise KeyError('invalid base_depth {} for resnet'.format(self.base_depth))
        self.norm_eval = False
        self.arch = [[int(i) for i in stage] for stage in self.arch.split('-')]
        self._make_stem_layer()
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
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                arch=arch,
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

    def _get_module_by_name(self, module_name):
        return self.__getattr__(module_name)

    def norm1(self, x):
        """Apply norm1."""
        return self._get_module_by_name(self.norm1_name)(x)

    def norm2(self, x):
        """Apply norm2."""
        return self._get_module_by_name(self.norm2_name)(x)

    def _make_stem_layer(self):
        """Make stem layer."""
        stem_channel = self.base_channel // 2
        norm_cfg = self.norm_cfg.copy()
        if self.norm_cfg.get('type') == 'GN':
            num_groups = norm_cfg.get('num_groups')
            norm_cfg['num_groups'] = int(num_groups / 2)
        norm_creator = LayerCreator(**norm_cfg)
        conv_creator = LayerCreator(**self.conv_cfg)
        self.conv1 = conv_creator.create_layer(3, stem_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1_name = norm_creator.get_name(magic_number=1)
        norm1 = norm_creator.create_layer(num_features=stem_channel)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv_creator.create_layer(stem_channel, self.base_channel, kernel_size=3,
                                               stride=2, padding=1, bias=False)
        self.norm2_name = norm_creator.get_name(magic_number=2)
        norm2 = norm_creator.create_layer(num_features=self.base_channel)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

    def _freeze_stages(self):
        """Freeze stages."""
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        if self.frozen_stages >= 1:
            self.norm2.eval()
            for m in [self.conv2, self.norm2]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward compute.

        :param x: input feature map
        :return: out feature map
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Train.

        :param mode: if train set mode True else False
        :type: bool
        """
        super(ResNetVariantDet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
