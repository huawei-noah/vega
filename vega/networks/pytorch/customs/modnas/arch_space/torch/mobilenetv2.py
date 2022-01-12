# -*- coding:utf-8 -*-

# This file is adapted from the MobileNetV2-pytorch library at
# https://github.com/Randl/MobileNetV2-pytorch

# 2020.6.29-Changed for Modular-NAS search space.
#         Huawei Technologies Co., Ltd. <linyunfeng5@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.

"""MobileNetV2 architectures."""

import math
from functools import partial
from collections import OrderedDict
import torch.nn as nn
from modnas.registry.construct import register as register_constructor
from modnas.registry.construct import DefaultMixedOpConstructor, DefaultSlotTraversalConstructor,\
    DefaultSlotArchDescConstructor
from modnas.registry.arch_space import register
from ..ops import get_same_padding
from ..slot import Slot, register_slot_builder


def round_filters(filters, width_coeff, divisor, min_depth=None):
    """Return rounded channel number."""
    multiplier = width_coeff
    if not multiplier:
        return filters
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coeff):
    """Return rounded repeat number."""
    multiplier = depth_coeff
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class MobileInvertedConv(nn.Sequential):
    """MobileNetV2 Inverted Residual Convolution."""

    def __init__(self, chn_in, chn_out, stride, expansion=6, kernel_size=3, padding=1, activation=nn.ReLU6):
        chn = chn_in * expansion
        nets = [] if chn_in == chn else [
            nn.Conv2d(chn_in, chn, kernel_size=1, bias=False),
            nn.BatchNorm2d(chn),
            activation(inplace=True),
        ]
        nets.extend([
            nn.Conv2d(chn, chn, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=chn),
            nn.BatchNorm2d(chn),
            activation(inplace=True),
            nn.Conv2d(chn, chn_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(chn_out)
        ])
        super().__init__(*nets)


class MobileInvertedResidualBlock(nn.Module):
    """MobileNetV2 Inverted Residual Block."""

    def __init__(self, chn_in, chn_out, stride=1, t=6, activation=nn.ReLU6):
        super(MobileInvertedResidualBlock, self).__init__()
        self.stride = stride
        self.t = t
        self.chn_in = chn_in
        self.chn_out = chn_out
        self.conv = Slot(_chn_in=chn_in, _chn_out=chn_out, _stride=stride, expansion=t, activation=activation)

    def forward(self, x):
        """Compute network output."""
        residual = x
        out = self.conv(x)
        if self.stride == 1 and self.chn_in == self.chn_out:
            out += residual
        return out


@register
class MobileNetV2(nn.Module):
    """MobileNetV2 Architecture Backbone."""

    def __init__(self,
                 cfgs,
                 chn_in=3,
                 n_classes=1000,
                 width_coeff=1.0,
                 depth_coeff=1.0,
                 resolution=None,
                 dropout_rate=0.2,
                 activation=nn.ReLU6):
        del resolution
        super(MobileNetV2, self).__init__()
        self.activation = activation
        self.n_classes = n_classes

        divisor = 8
        self.t = [cfg[0] for cfg in cfgs]
        self.c = [round_filters(cfg[1], width_coeff, divisor) for cfg in cfgs]
        self.n = [round_repeats(cfg[2], depth_coeff) for cfg in cfgs]
        self.s = [cfg[3] for cfg in cfgs]
        self.conv_first = nn.Sequential(
            nn.Conv2d(chn_in, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1),
            nn.BatchNorm2d(self.c[0]),
            self.activation(inplace=True),
        )
        self.bottlenecks = self._make_bottlenecks()

        self.last_conv_out_ch = round_filters(1280, width_coeff, divisor)
        self.conv_last = nn.Sequential(
            nn.Conv2d(self.c[-1], self.last_conv_out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.last_conv_out_ch),
            self.activation(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.fc = nn.Linear(self.last_conv_out_ch, self.n_classes)

    def _make_stage(self, chn_in, chn_out, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "MobileInvertedResidualBlock_{}".format(stage)
        for i in range(n):
            # First module is the only one utilizing stride
            s = stride if i == 0 else 1
            name = stage_name + "_{}".format(i)
            module = MobileInvertedResidualBlock(chn_in=chn_in,
                                                 chn_out=chn_out,
                                                 stride=s,
                                                 t=t,
                                                 activation=self.activation)
            modules[name] = module
            chn_in = chn_out
        return nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"
        for i in range(0, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(chn_in=self.c[i],
                                      chn_out=self.c[i + 1],
                                      n=self.n[i + 1],
                                      stride=self.s[i + 1],
                                      t=self.t[i + 1],
                                      stage=i)
            modules[name] = module
        return nn.Sequential(modules)

    def forward(self, x):
        """Compute network output."""
        x = self.conv_first(x)
        x = self.bottlenecks(x)
        x = self.conv_last(x)

        # average pooling layer
        x = self.avgpool(x)
        x = self.dropout(x)

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mbv2_predefined_convert_fn(slot):
    """MobileNetV2 predefined converter."""
    return MobileInvertedConv(slot.chn_in, slot.chn_out, stride=slot.stride, **slot.kwargs)


@register_constructor
class MobileNetV2PredefinedConstructor(DefaultSlotTraversalConstructor):
    """MobileNetV2 predefined Constructor."""

    def convert(self, slot):
        """Convert Slot to MobileNetV2 predefined module."""
        return mbv2_predefined_convert_fn(slot)


@register_constructor
class MobileNetV2SearchConstructor(DefaultMixedOpConstructor):
    """MobileNetV2 mixed operator search space Constructor."""

    def __init__(self, *args, fix_first=True, add_zero_op=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.fix_first = fix_first
        self.add_zero_op = add_zero_op
        self.first = True
        self.predefined_fn = mbv2_predefined_convert_fn

    def convert(self, slot):
        """Convert Slot to MixedOp."""
        if self.fix_first and self.first:
            ent = self.predefined_fn(slot)
            self.first = False
        else:
            cands = self.candidates[:]
            if self.add_zero_op and slot.stride == 1 and slot.chn_in == slot.chn_out:
                self.candidates.append('NIL')
            ent = super().convert(slot)
            self.candidates = cands
        return ent


@register_constructor
class MobileNetV2ArchDescConstructor(DefaultSlotArchDescConstructor):
    """MobileNetV2 archdesc Constructor."""

    def __init__(self, *args, fix_first=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.fix_first = fix_first
        self.first = True
        self.predefined_fn = mbv2_predefined_convert_fn

    def convert(self, slot):
        """Convert Slot to module from archdesc."""
        if self.fix_first and self.first:
            ent = self.predefined_fn(slot)
            self.first = False
            self.get_next_desc()
        else:
            ent = super().convert(slot)
        return ent


_mbv2_ori_cfgs = [
    # t, c, n, s,
    [0, 32, 1, 2],
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1]
]


_mbv2_gpu_cfgs = [
    # t, c, n, s,
    [0, 32, 1, 2],
    [1, 16, 1, 1],
    [6, 24, 4, 2],
    [6, 40, 4, 2],
    [6, 80, 4, 2],
    [6, 96, 4, 1],
    [6, 192, 4, 2],
    [6, 320, 1, 1]
]


def mobilenetv2(cfgs=None, cifar=False, **kwargs):
    """Return MobileNetV2 model."""
    if cfgs is None:
        cfgs = _mbv2_ori_cfgs
    if cifar:
        cfgs[0][3] = 1
        cfgs[6][3] = 1
    return MobileNetV2(cfgs=cfgs, **kwargs)


for cifar_format in [True, False]:
    img = 'CIFAR' if cifar_format else 'ImageNet'
    register(partial(mobilenetv2, cifar=cifar_format), '{}_MobileNetV2'.format(img))
    register(partial(mobilenetv2, cfgs=_mbv2_gpu_cfgs, cifar=cifar_format), '{}_MobileNetV2_GPU'.format(img))


kernel_sizes = [3, 5, 7, 9]
expand_ratios = [1, 3, 6, 9]
for k in kernel_sizes:
    for e in expand_ratios:
        p = get_same_padding(k)
        builder = partial(MobileInvertedConv, expansion=e, kernel_size=k, padding=p)
        register_slot_builder(builder, 'MB{}E{}'.format(k, e), 'i1o1s2')
