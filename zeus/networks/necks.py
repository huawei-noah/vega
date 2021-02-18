# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ResNet models for detection."""
from zeus.common.class_factory import ClassFactory, ClassType
from zeus.modules.module import Module
from zeus.modules.operators import ops
from zeus.modules.connections import Sequential


class BasicBlock(Module):
    """This is the class of BasicBlock block for ResNet."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 style='pytorch', with_cp=False):
        """Init BasicBlock."""
        super(BasicBlock, self).__init__()
        self.expansion = 1
        self.norm1 = ops.BatchNorm2d(planes)
        self.norm2 = ops.BatchNorm2d(planes)
        self.conv1 = ops.Conv2d(inplanes, planes, 3, stride=stride, padding=dilation,
                                dilation=dilation, bias=False)
        self.conv2 = ops.Conv2d(
            planes, planes, 3, padding=1, bias=False)
        self.relu = ops.Relu(inplace=True)
        conv_layer = ops.Conv2d(inplanes, planes,
                                kernel_size=1, stride=stride, bias=False)
        norm_layer = ops.BatchNorm2d(planes)
        downsample = Sequential(conv_layer, norm_layer)
        self.downsample = downsample
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        assert not with_cp

    def call(self, x):
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


class Bottleneck(Module):
    """This is the class of Bottleneck block for ResNet."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 style='pytorch', with_cp=False):
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
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1 = ops.BatchNorm2d(planes)
        self.norm2 = ops.BatchNorm2d(planes)
        self.norm3 = ops.BatchNorm2d(planes * self.expansion)
        self.conv1 = ops.Conv2d(
            inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False)
        self.with_modulated_dcn = False
        self.conv2 = ops.Conv2d(planes, planes, kernel_size=3, stride=self.conv2_stride,
                                padding=dilation, dilation=dilation, bias=False, )
        self.conv3 = ops.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.relu = ops.Relu(inplace=True)
        self.downsample = downsample

    def call(self, x):
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
            if not self.with_cp:
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
            out = self.conv3(out)
            out = self.norm3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out

        out = _inner_forward(x)
        out = self.relu(out)
        return out


def make_res_layer_from_code(block, inplanes, planes, blocks, stride=1, dilation=1,
                             style='pytorch', with_cp=False, code=None):
    """Make res layer from code."""
    if code is None:
        return make_res_layer(block, inplanes, planes, blocks, stride, dilation, style, with_cp)

    strides = map(int, code)
    layers = []
    for stride in strides:
        layers.append(block(inplanes=inplanes, planes=planes, stride=stride, dilation=dilation,
                            style=style, with_cp=with_cp))
        inplanes = planes * block.expansion
    return Sequential(*layers)


def make_res_layer(block, inplanes, planes, blocks, stride=1, dilation=1, style='pytorch', with_cp=False):
    """Build resnet layer."""
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        conv_layer = ops.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                                bias=False)
        norm_layer = ops.BatchNorm2d(
            planes * block.expansion)
        downsample = Sequential(conv_layer, norm_layer)
    layers = []
    layers.append(block(inplanes=inplanes, planes=planes, stride=stride, dilation=dilation,
                        downsample=downsample, style=style, with_cp=with_cp))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes=inplanes, planes=planes, stride=1, dilation=dilation,
                            style=style, with_cp=with_cp))
    return Sequential(*layers)


class ConvModule(ops.Module):
    """Conv Module with Normalization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias='auto',
                 activation='relu', inplace=True, activate_last=True):
        """Init Conv Module with Normalization."""
        super(ConvModule, self).__init__()
        self.activation = activation
        self.inplace = inplace
        self.activate_last = activate_last
        self.with_norm = True
        self.with_activatation = activation is not None
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias
        self.conv = ops.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, bias=bias)
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
            self.norm = ops.BatchNorm2d(norm_channels)
        if self.with_activatation:
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = ops.Relu(inplace=inplace)

    def call(self, x, activate=True, norm=True):
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


@ClassFactory.register(ClassType.NETWORK)
class FPN(Module):
    """FPN."""

    def __init__(self, in_channels=[64, 128, 256, 512], out_channels=256, num_outs=5,
                 activation=None, start_level=0, end_level=-1,
                 add_extra_convs=None, extra_convs_on_inputs=None, relu_before_extra_convs=None):
        """Init FPN.

        :param desc: config dict
        """
        super(FPN, self).__init__()
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.relu_before_extra_convs = relu_before_extra_convs
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert self.num_outs >= self.num_ins - self.start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(self.in_channels)
            assert self.num_outs == end_level - self.start_level
        self.lateral_convs = ops.MoudleList()
        self.fpn_convs = ops.MoudleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i], out_channels, 1, activation=activation, inplace=False)
            fpn_conv = ConvModule(
                out_channels, out_channels, 3, padding=1, activation=activation, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        extra_levels = self.num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels, out_channels, 3, stride=2, padding=1,
                                            activation=activation, inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def call(self, inputs):
        """Forward compute.

        :param inputs: input feature map
        :return: tuple of feature map
        """
        # assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level])
                    for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += ops.InterpolateScale(
                scale_factor=2, mode='nearest')(laterals[i])
        outs = [self.fpn_convs[i](laterals[i])
                for i in range(used_backbone_levels)]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(ops.MaxPool2d(1, stride=2)(outs[-1]))
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](ops.Relu()(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return {idx: out for idx, out in enumerate(outs)}
