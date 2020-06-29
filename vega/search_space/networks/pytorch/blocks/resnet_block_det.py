# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined ResNet Blocks For Detection."""
import torch.nn as nn
from .conv_ws import ConvWS2d

norm_cfg_dict = {'BN': ('bn', nn.BatchNorm2d),
                 'GN': ('gn', nn.GroupNorm)}

conv_cfg_dict = {'Conv': nn.Conv2d,
                 'ConvWS': ConvWS2d}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """Generate 3x3 convolution layer."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """Generate 1x1 convolution layer."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """This is the class of BasicBlock block for ResNet.

    :param inplanes: input feature map channel num
    :type inplanes: int
    :param planes: output feature map channel num
    :type planes: int
    :param stride: stride
    :type stride: int
    :param dilation: dilation
    :type dilation: int
    :param downsample: downsample
    :param style: style, "pytorch" mean 3x3 conv layer and "caffe" mean the 1x1 conv layer.
    :type style: str
    :param with_cp: with cp
    :type with_cp: bool
    :param conv_cfg: conv config
    :type conv_cfg: dict
    :param norm_cfg: norm config
    :type norm_cfg: dict
    :param dcn: deformable conv network
    :param gcb: gcb
    :param gen_attention: gen attention
    """

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg={"type": 'Conv'},
                 norm_cfg={"type": 'BN'}):
        """Init BasicBlock."""
        super(BasicBlock, self).__init__()
        self.expansion = 1
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        requires_grad = self.norm_cfg['requires_grad'] if 'requires_grad' in self.norm_cfg else False
        if self.norm_cfg['type'] == 'BN':
            self.norm1 = norm_cfg_dict[self.norm_cfg['type']][1](planes)
            self.norm2 = norm_cfg_dict[self.norm_cfg['type']][1](planes)
        else:
            self.norm1 = norm_cfg_dict[self.norm_cfg['type']][1](num_channels=planes)
            self.norm2 = norm_cfg_dict[self.norm_cfg['type']][1](num_channels=planes)
        if requires_grad:
            for param in self.norm1.parameters():
                param.requires_grad = requires_grad
            for param in self.norm2.parameters():
                param.requires_grad = requires_grad
        self.norm1_name = norm_cfg_dict[self.norm_cfg['type']][0] + '_1'
        self.norm2_name = norm_cfg_dict[self.norm_cfg['type']][0] + '_2'
        self.conv1 = conv_cfg_dict[self.conv_cfg['type']](inplanes, planes, 3, stride=stride, padding=dilation,
                                                          dilation=dilation, bias=False)
        self.add_module(self.norm1_name, self.norm1)
        self.conv2 = conv_cfg_dict[self.conv_cfg['type']](planes, planes, 3, padding=1, bias=False, )
        self.add_module(self.norm2_name, self.norm2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        assert not with_cp

    def forward(self, x):
        """Forward compute.

        :param x: input feature map
        :type x: torch.Tensor
        :return: output feature map
        :rtype: torch.Tensor
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
    """This is the class of Bottleneck block for ResNet.

    :param inplanes: input feature map channel num
    :type inplanes: int
    :param planes: output feature map channel num
    :type planes: int
    :param stride: stride
    :type stride: int
    :param dilation: dilation
    :type dilation: int
    :param downsample: downsample
    :param style:  style, "pytorch" mean 3x3 conv layer and "caffe" mean the 1x1 conv layer.
    :type style: str
    :param with_cp: with cp
    :type with_cp: bool
    :param conv_cfg: conv config
    :type conv_cfg: dict
    :param norm_cfg: norm config
    :type norm_cfg: dict
    :param dcn: deformable conv network
    :param gcb: gcb
    :param gen_attention: gen attention
    """

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg={"type": 'Conv'},
                 norm_cfg={"type": 'BN'}):
        """Init Bottleneck."""
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.expansion = 4
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        requires_grad = self.norm_cfg['requires_grad'] if 'requires_grad' in self.norm_cfg else False
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        if self.norm_cfg['type'] == 'BN':
            self.norm1 = norm_cfg_dict[self.norm_cfg['type']][1](planes)
            self.norm2 = norm_cfg_dict[self.norm_cfg['type']][1](planes)
            self.norm3 = norm_cfg_dict[self.norm_cfg['type']][1](planes * self.expansion)
        else:
            self.norm1 = norm_cfg_dict[self.norm_cfg['type']][1](num_channels=planes)
            self.norm2 = norm_cfg_dict[self.norm_cfg['type']][1](num_channels=planes)
            self.norm3 = norm_cfg_dict[self.norm_cfg['type']][1](planes * self.expansion)
        if requires_grad:
            for param in self.norm1.parameters():
                param.requires_grad = requires_grad
            for param in self.norm2.parameters():
                param.requires_grad = requires_grad
            for param in self.norm3.parameters():
                param.requires_grad = requires_grad
        self.norm1_name = norm_cfg_dict[self.norm_cfg['type']][0] + '_1'
        self.norm2_name = norm_cfg_dict[self.norm_cfg['type']][0] + '_2'
        self.norm3_name = norm_cfg_dict[self.norm_cfg['type']][0] + '_3'
        self.conv1 = conv_cfg_dict[self.conv_cfg['type']](inplanes, planes, kernel_size=1, stride=self.conv1_stride,
                                                          bias=False, )
        self.add_module(self.norm1_name, self.norm1)
        self.with_modulated_dcn = False
        self.conv2 = conv_cfg_dict[self.conv_cfg['type']](planes, planes, kernel_size=3, stride=self.conv2_stride,
                                                          padding=dilation, dilation=dilation, bias=False, )
        self.add_module(self.norm2_name, self.norm2)
        self.conv3 = conv_cfg_dict[self.conv_cfg['type']](planes, planes * self.expansion, kernel_size=1,
                                                          bias=False)
        self.add_module(self.norm3_name, self.norm3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """Forward compute.

        :param x: input feature map
        :type x: torch.Tensor
        :return: out feature map
        :rtype: torch.Tensor
        """

        def _inner_forward(x):
            """Inner forward.

            :param x: input feature map
            :type x: torch.Tensor
            :return: out feature map
            :rtype: torch.Tensor
            """
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            if not self.with_dcn:
                out = self.conv2(out)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out)
                offset = offset_mask[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = self.norm2(out)
            out = self.relu(out)
            if self.with_gen_attention:
                out = self.gen_attention_block(out)
            out = self.conv3(out)
            out = self.norm3(out)
            if self.with_gcb:
                out = self.context_block(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out

        out = _inner_forward(x)
        out = self.relu(out)
        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg={"type": 'Conv'},
                   norm_cfg={"type": 'BN'}):
    """Build resnet layer.

    :param block: Block function
    :type blocks: Bottleneck or BasicBlock
    :param inplanes: input planes
    :type inplanes: int
    :param planes: output planes
    :type planes: int
    :param blocks: num of blocks
    :type block: int
    :param stride: stride of convolution
    :type stride: int
    :param dilation: num of dilation
    :type dilation: int
    :param style: style of bottle neck connect
    :type style: str
    :param with_cp: if with conv cp
    :type with_cp: bool
    :param conv_cfg: convolution config
    :type conv_cfg: dict
    :param norm_cfg: normalization config
    :type norm_cfg: dict
    :return: resnet layer
    :rtype: nn.Sequential
    """
    downsample = None
    requires_grad = norm_cfg['requires_grad'] if 'requires_grad' in norm_cfg else False
    if stride != 1 or inplanes != planes * block.expansion:
        conv_layer = conv_cfg_dict[conv_cfg['type']](inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                                                     bias=False)
        norm_layer = norm_cfg_dict[norm_cfg['type']][1](planes * block.expansion)
        if requires_grad:
            for param in norm_layer.parameters():
                param.requires_grad = requires_grad
        downsample = nn.Sequential(conv_layer, norm_layer)
    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, ))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))
    return nn.Sequential(*layers)
