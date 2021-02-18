# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Conv Module with Normalization."""
import warnings
import torch.nn as nn
from .conv_ws import ConvWS2d

conv_cfg_dict = {
    'Conv': nn.Conv2d,
    'ConvWS': ConvWS2d,
}

norm_cfg_dict = {'BN': ('bn', nn.BatchNorm2d),
                 'GN': ('gn', nn.GroupNorm)}


class ConvModule(nn.Module):
    """Conv Module with Normalization."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg={"type": 'Conv'},
                 norm_cfg={"type": 'BN'},
                 activation='relu',
                 inplace=True,
                 activate_last=True):
        """Init Conv Module with Normalization.

        :param in_channels: input channels
        :type in_channels: int
        :param out_channels: output channels
        :type out_channels: int
        :param kernel_size: kernel size
        :type kernel_size: int
        :param stride: convolution stride
        :type stride: int
        :param padding: num of padding
        :type padding: int
        :param dilation: num of dilation
        :type dilation: int
        :param groups: num of group
        :type groups: int
        :param bias: type of bias
        :type bias: str
        :param conv_cfg: config of convolution layer
        :type conv_cfg: dict
        :param norm_cfg: config of normalization layer
        :type norm_cfg: dict
        :param activation: activation after module
        :type activation: str
        :param inplace: if conv inplace
        :type inplace: bo'
        :param activate_last: if last activate
        :type activate_last: bool
        """
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.activate_last = activate_last
        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias
        self.conv = conv_cfg_dict[self.conv_cfg['type']](
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups
        if self.with_norm:
            norm_channels = out_channels if self.activate_last else in_channels
            requires_grad = self.norm_cfg['requires_grad'] if 'requires_grad' in self.norm_cfg else False
            self.norm = norm_cfg_dict[self.norm_cfg['type']][1](norm_channels)
            self.norm_name = norm_cfg_dict[self.norm_cfg['type']][0]
            if requires_grad:
                for param in self.norm.parameters():
                    param.requires_grad = requires_grad
            self.add_module(self.norm_name, self.norm)
        if self.with_activatation:
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)
        self.init_weight()

    def init_weight(self):
        """Init weight of Conv Module with Normalization."""
        nonlinearity = 'relu' if self.activation is None else self.activation
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity=nonlinearity)
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        if self.with_norm:
            nn.init.constant_(self.norm.weight, 1)
        if hasattr(self.with_norm, 'bias') and self.with_norm.bias is not None:
            nn.init.constant_(self.with_norm.bias, 0)

    def forward(self, x, activate=True, norm=True):
        """Forward compute of Conv Module with Normalization."""
        if self.activate_last:
            x = self.conv(x)
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
        else:
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
            x = self.conv(x)
        return x
