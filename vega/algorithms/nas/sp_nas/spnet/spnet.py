# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Modules for SpNet."""

import logging
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from mmdet.models.utils import build_conv_layer, build_norm_layer

# customize by SpNas
from .backbone_tools import match_name, remove_layers, load_checkpoint, dirac_init
import os.path as osp
import torch
from collections import OrderedDict
import torch.nn.functional as F
import math
from mmdet.models.registry import BACKBONES


class BasicBlock(nn.Module):
    """Class of BasicBlock block for ResNet.

    :param inplanes: input feature map channel num
    :type inplanes: int

    :param planes: output feature map channel num
    :type planes: int

    :param stride: stride
    :type stride: int

    :param dilation: dilation
    :type dilation: int

    :param downsample: downsample

    :param style: style,
    "pytorch" mean the stride-two layer is the 3x3 conv layer and
    "caffe" mean the stride-two layer is the first 1x1 conv layer.
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

    :param groups: group num
    :type groups: int

    :param base_width: base width each group
    :type base_width: int

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
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None,
                 groups=1,
                 base_width=4):
        super(BasicBlock, self).__init__()
        assert dcn is None, "Not implemented yet."
        assert gen_attention is None, "Not implemented yet."
        assert gcb is None, "Not implemented yet."
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        """Get norm layer 1.

        :return: norm layer 1
        :rtype: torch.nn.module
        """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """Get norm layer 2.

        :return: norm layer 2
        :rtype: torch.nn.module
        """
        return getattr(self, self.norm2_name)

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
    """Class of Bottleneck block for ResNet.

    :param inplanes: input feature map channel num
    :type inplanes: int

    :param planes: output feature map channel num
    :type planes: int

    :param stride: stride
    :type stride: int

    :param dilation: dilation
    :type dilation: int

    :param downsample: downsample

    :param style: style,
    "pytorch" mean the stride-two layer is the 3x3 conv layer and
    "caffe" mean the stride-two layer is the first 1x1 conv layer
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

    :param groups: group num
    :type groups: int

    :param base_width: base width each group
    :type base_width: int

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
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None,
                 groups=1,
                 base_width=4):
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert gcb is None or isinstance(gcb, dict)
        assert gen_attention is None or isinstance(gen_attention, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.gcb = gcb
        self.with_gcb = gcb is not None
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes * (base_width / 64)) * groups

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                groups=groups,
                bias=False)
        else:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            self.deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                self.deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2 = conv_op(
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                groups=groups,
                deformable_groups=self.deformable_groups,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            width,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_gcb:
            gcb_inplanes = planes * self.expansion
            self.context_block = ContextBlock(
                inplanes=gcb_inplanes,
                **gcb
            )

        # gen_attention
        if self.with_gen_attention:
            self.gen_attention_block = GeneralizedAttention(
                planes, **gen_attention)

    @property
    def norm1(self):
        """Get norm layer 1.

        :return: norm layer 1
        :rtype: torch.nn.module
        """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """Get norm layer 2.

        :return: norm layer 2
        :rtype: torch.nn.module
        """
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """Get norm layer 3.

        :return: norm layer 3
        :rtype: torch.nn.module
        """
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward compute.

        :param x: input feature map
        :type x: torch.Tensor
        :return: out feature map
        :rtype: torch.Tensor
        """

        def _inner_forward(x):
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

            # if self.with_gen_attention:
            #     out = self.gen_attention_block(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_gcb:
                out = self.context_block(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   arch,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   gcb=None,
                   gen_attention=None,
                   gen_attention_blocks=[],
                   groups=1,
                   base_width=4):
    """Make resnet layer.

    :param block: block function

    :param inplanes: input feature map channel num
    :type inplanes: int

    :param planes: output feature map channel num
    :type planes: int

    :param arch: model arch
    :type arch: str

    :param stride: stride
    :type stride: int

    :param dilation: dilation
    :type dilation: int

    :param style: style
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

    :param gen_attention_blocks: gen attention block

    :param groups: groups
    :type planes: int

    :param base_width: base width
    :type planes: int

    :return: layer
    """
    layers = []
    for i, layer_type in enumerate(arch):
        downsample = None
        stride = stride if i == 0 else 1
        if layer_type == 2:
            planes *= 2
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1])
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
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                groups=groups,
                base_width=base_width,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None))
        inplanes = planes * block.expansion
    return nn.Sequential(*layers)


def make_connection(inplanes,
                    planes,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN')):
    """Make connection layer.

    :param inplanes: channels of current stage output
    :type inplanes: int

    :param planes: channels of current stage input
    :type planes: int

    :param conv_cfg: conv config
    :type conv_cfg: dict

    :param norm_cfg: norm config
    :type norm_cfg: dict

    :return: layer
    """
    layer = nn.Sequential(
        build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=1,
            bias=False),
        build_norm_layer(norm_cfg, planes)[1],
    )
    return layer


@BACKBONES.register_module
class SpNet(nn.Module):
    """Class of SpNet backbone.

    :param pretrained_arch: base_depth of resnet, from {18, 34, 50, 101, 152}.
    :type pretrained_arch: str

    :param pretrained_arch: base_depth of resnet, from {18, 34, 50, 101, 152}.
    :type pretrained_arch: str

    :param num_stages: Resnet stages, normally 4.
    :type num_steags: int

    :param strides: Strides of the first block of each stage.
    :type strides: tuple

    :param dilations: Dilation of each stage.
    :type dilations: tuple

    :param out_indices: Output from which stages.
    :type out_indices: tuple

    :param style: conv layer style.
    "pytorch" mean the stride-two layer is the 3x3 conv layer,
    "caffe" mean the stride-two layer is the first 1x1 conv layer.
    :type style: str

    :param frozen_stages: Stages to be frozen (stop grad and set eval mode)
    -1 means not freezing any parameters.
    :type frozen_stages: int

    :param norm_cfg: dictionary to construct and config norm layer.
    :type norm_cfg: dict

    :param norm_eval : Whether to set norm layers to eval mode, namely
    :type norm_eval: bool

    :param with_cp: Use checkpoint or not. Using checkpoint will save
    some memory while slowing down the training speed.
    :type with_cp: bool

    :param zero_init_residual: whether to use zero init for last norm layer
    in resblocks to let them behave as identity.
    :type zero_init_residual: bool
    """

    def __init__(self,
                 pretrained_arch,
                 layers_arch,
                 mb_arch,
                 arch_block='Bottleneck',
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True,
                 groups=1,
                 base_width=4,
                 base_channel=64,
                 reignition=False):
        super(SpNet, self).__init__()
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.gen_attention = gen_attention
        self.stage_with_gen_attention = stage_with_gen_attention
        self.gcb = gcb
        self.stage_with_gcb = stage_with_gcb
        if gcb is not None:
            assert len(stage_with_gcb) == num_stages
        self.zero_init_residual = zero_init_residual
        self.groups = groups
        self.base_width = base_width
        self.base_channel = base_channel
        # identity arch settings
        self.layer_arch = layers_arch.split('-')
        arch_ = [[int(i) for i in stage] for stage in self.layer_arch]
        self.mb_arch = [int(m) for m in mb_arch.split('-')]
        assert len(strides) == len(dilations) == num_stages, "Out of stage!"
        assert num_stages == len(self.mb_arch) == len(self.layer_arch), "Out of stage!"
        if arch_block:
            assert arch_block in ['Bottleneck', 'BasicBlock'], "Error Block type."
            self.block = Bottleneck if arch_block == 'Bottleneck' else BasicBlock
        self.pretrained_arch = pretrained_arch

        self.reignition = reignition
        if self.reignition:
            self._make_stem_layer2(self.base_channel)
        else:
            self._make_stem_layer()
        self.res_layers = []

        total_expand = 0
        inplanes = planes = self.base_channel
        for i, arch in enumerate(arch_):
            num_expand = arch.count(2)
            total_expand += num_expand
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            gcb = self.gcb if self.stage_with_gcb[i] else None
            res_layer = make_res_layer(
                self.block,
                inplanes,
                planes,
                arch,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=self.gen_attention,
                gen_attention_blocks=self.stage_with_gen_attention[i],
                groups=self.groups,
                base_width=self.base_width)

            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            planes = self.base_channel * 2 ** total_expand

            # MB branch
            if self.mb_arch[i] > 0:
                connect_layer = make_connection(planes * self.block.expansion, inplanes)
                for m in range(self.mb_arch[i]):
                    self.add_module('mb{}_{}'.format(i + 1, m), res_layer)
                    self.add_module('connect{}_{}'.format(i + 1, m), connect_layer)

            inplanes = planes * self.block.expansion

        self._freeze_stages()
        self.feat_dim = self.block.expansion * self.base_channel * 2 ** total_expand

    @property
    def norm1(self):
        """Get norm for norm 1."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """Get norm for norm 1."""
        return getattr(self, self.norm2_name)

    def _make_stem_layer(self):
        """Make stem layer 1."""
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            self.base_channel,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, self.base_channel, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_stem_layer2(self, base_channel=64):
        """Make stem layer 2, after reignite."""
        self.conv1 = nn.Conv2d(
            3,
            base_channel // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, base_channel // 2, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = nn.Conv2d(
            base_channel // 2,
            base_channel,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, base_channel, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

    def _freeze_stages(self):
        """Freeze stages."""
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):  # noqa: C901
        """Init weights."""
        if isinstance(pretrained, str):
            # initialize all layer with Dirac delta function to keep the identity of the input
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    dirac_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 0)

            # match imagenet pretrained model
            if not osp.isfile(pretrained):
                raise IOError('{} is not a checkpoint file'.format(pretrained))
            checkpoint = torch.load(pretrained)
            # get state_dict from checkpoint
            if isinstance(checkpoint, OrderedDict):
                checkpoint = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))

            own_state = self.state_dict()

            # when len(arch) not equal with len(pretrained arch)
            len_dis = len('-'.join(self.layer_arch)) - len(self.pretrained_arch)
            if len_dis > 0:
                remove_name, bn_layers = remove_layers(self.layer_arch, self.pretrained_arch, len_dis)
                for k in self.state_dict().keys():
                    for r_name in remove_name:
                        if k.find(r_name) >= 0:
                            del own_state[k]
                # turn off eval mode for all newly added bn layers
                modules = [layer[1] for layer in self.named_modules() if layer[0] in bn_layers]
                for m in modules:
                    m.eval_mode = False

            pretrain_to_own = match_name(own_state.keys(), checkpoint.keys())

            mb_mapping = {}
            for mlayer in own_state.keys():
                if mlayer.find('mb') >= 0:
                    layer_n = mlayer.split('.')
                    layer_n[0] = mlayer.replace('mb', 'layer').split('_')[0]
                    layer_n = '.'.join(layer_n)
                    if layer_n in mb_mapping:
                        mb_mapping[layer_n].append(mlayer)
                    else:
                        mb_mapping[layer_n] = [mlayer]

            logger = logging.getLogger()
            load_checkpoint(self, pretrained, pretrain_to_own, logger=logger, mb_mapping=mb_mapping)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward compute.

        :param x: input feature map
        :type x: torch.Tensor

        :return: out feature map
        :rtype: torch.Tensor
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if self.reignition:
            x = self.conv2(x)
            x = self.norm2(x)
            x = self.relu(x)
        else:
            x = self.maxpool(x)
        outs = []

        x_mb_list = {}
        for i, layer_name in enumerate(self.res_layers):
            temp = 0
            if self.mb_arch[i] > 0:
                x_mb_new = {}
                for mb_i in range(self.mb_arch[i] - 1, -1, -1):
                    mb_layer = getattr(self, 'mb' + str(i + 1) + '_' + str(mb_i))
                    connect_layer = getattr(self, 'connect' + str(i + 1) + '_' + str(mb_i))
                    if len(x_mb_list) == 0:
                        x_mb = x
                    elif mb_i in x_mb_list.keys():
                        x_mb = x_mb_list[mb_i]
                    else:
                        x_mb = x_mb_list[len(x_mb_list) - 1]
                    x_mb = mb_layer(x_mb + temp)

                    # record output for next stage
                    if i + 1 < len(self.res_layers) and mb_i in range(self.mb_arch[i + 1] - 1, -1, -1):
                        x_mb_new[mb_i] = x_mb

                    temp = F.interpolate(
                        connect_layer(x_mb), size=x.size()[2:], mode='nearest')

                x_mb_list = x_mb_new
                del x_mb_new

            # main branch
            res_layer = getattr(self, layer_name)
            x = res_layer(x + temp)

            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Train.

        :param mode: if train
        :type mode: bool
        """
        super(SpNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm) and getattr(m, 'eval_mode', True):
                    m.eval()
