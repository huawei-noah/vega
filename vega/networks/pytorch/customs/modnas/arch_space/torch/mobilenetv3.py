# -*- coding:utf-8 -*-

# This file is adapted from the mobilenetv3.pytorch library at
# https://github.com/d-li14/mobilenetv3.pytorch

# 2020.6.29-Changed for Modular-NAS search space.
#         Huawei Technologies Co., Ltd. <linyunfeng5@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.

"""MobileNetV3 architectures."""

import torch.nn as nn
import torch.nn.functional as F
from modnas.registry.construct import DefaultMixedOpConstructor, DefaultSlotTraversalConstructor,\
    DefaultSlotArchDescConstructor
from modnas.registry.construct import register as register_constructor
from modnas.registry.arch_space import register
from ..slot import Slot, register_slot_ccs

for ks in [3, 5, 7, 9]:
    for exp in [1, 3, 6, 9]:
        register_slot_ccs(lambda C_in, C_out, S, use_se=0, use_hs=0, k=ks, e=exp: MobileInvertedConvV3(
                          C_in, C_out, S, C_in * e, k, use_se, use_hs), 'M3B{}E{}'.format(ks, exp))
        register_slot_ccs(lambda C_in, C_out, S, k=ks, e=exp: MobileInvertedConvV3(C_in, C_out, S, C_in * e, k, 0, 1),
                          'M3B{}E{}H'.format(ks, exp))
        register_slot_ccs(lambda C_in, C_out, S, k=ks, e=exp: MobileInvertedConvV3(C_in, C_out, S, C_in * e, k, 1, 0),
                          'M3B{}E{}S'.format(ks, exp))
        register_slot_ccs(lambda C_in, C_out, S, k=ks, e=exp: MobileInvertedConvV3(C_in, C_out, S, C_in * e, k, 1, 1),
                          'M3B{}E{}SH'.format(ks, exp))


def _make_divisible(v, divisor, min_value=None):
    """Make channel divisible.

    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation function."""

    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """Compute network output."""
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class HardSwish(nn.Module):
    """Hard Swish activation function."""

    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """Compute network output."""
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer."""

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        chn = _make_divisible(channel // reduction, divisor=8)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channel, chn, 1, 1, 0), nn.ReLU(inplace=True),
                                 nn.Conv2d(chn, channel, 1, 1, 0), HardSigmoid())

    def forward(self, x):
        """Compute network output."""
        return x * self.net(x)


def conv_3x3_bn(chn_in, chn_out, stride, kernel_size, use_se, use_hs):
    """Return the first convolution layer."""
    del use_se
    return nn.Sequential(
        nn.Conv2d(chn_in, chn_out, kernel_size, stride, kernel_size // 2, bias=False),
        nn.BatchNorm2d(chn_out),
        HardSwish() if use_hs else nn.ReLU(inplace=True),
    )


class MobileInvertedConvV3(nn.Sequential):
    """MobileNetV3 Inverted Residual Convolution."""

    def __init__(self, chn_in, chn_out, stride, chn, kernel_size, use_se=0, use_hs=0):
        nets = []
        if chn_in != chn:
            nets.extend([
                nn.Conv2d(chn_in, chn, 1, 1, 0, bias=False),
                nn.BatchNorm2d(chn),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
            ])
        nets.extend([
            nn.Conv2d(chn, chn, kernel_size, stride, (kernel_size - 1) // 2, groups=chn, bias=False),
            nn.BatchNorm2d(chn),
            HardSwish() if use_hs else nn.ReLU(inplace=True),
        ])
        if use_se:
            nets.append(SELayer(chn))
        nets.extend([
            nn.Conv2d(chn, chn_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chn_out),
        ])
        super().__init__(*nets)


class MobileInvertedResidualBlock(nn.Module):
    """MobileNetV3 Inverted Residual Block."""

    def __init__(self, chn_in, chn, chn_out, kernel_size, stride, use_se, use_hs):
        super(MobileInvertedResidualBlock, self).__init__()
        if stride not in [1, 2]:
            raise ValueError('unknown stride: %s' % repr(stride))
        self.identity = stride == 1 and chn_in == chn_out
        self.conv = Slot(_chn_in=chn_in,
                         _chn_out=chn_out,
                         _stride=stride,
                         chn=chn,
                         kernel_size=kernel_size,
                         use_se=use_se,
                         use_hs=use_hs)

    def forward(self, x):
        """Compute network output."""
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


@register
class MobileNetV3(nn.Module):
    """MobileNetV3 base architecture."""

    def __init__(self, cfgs, mode, chn_in=3, n_classes=1000, width_mult=1., dropout_rate=0.2):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        if mode not in ['large', 'small']:
            raise ValueError('unknown mode: %s' % repr(mode))
        block = MobileInvertedResidualBlock
        # building layers
        layers = []
        for i, (k, exp_size, c, use_se, use_hs, s) in enumerate(self.cfgs):
            chn_out = _make_divisible(c * width_mult, 8)
            if i == 0:
                # building first layer
                layers.append(conv_3x3_bn(chn_in, chn_out, s, k, use_se, use_hs))
            else:
                # building inverted residual blocks
                layers.append(block(chn_in, exp_size, chn_out, k, s, use_se, use_hs))
            chn_in = chn_out
            last_chn = exp_size
        self.features = nn.Sequential(*layers)
        # building last several layers
        last_chn = _make_divisible(last_chn * width_mult, 8)
        self.conv = nn.Sequential(nn.Conv2d(chn_in, last_chn, 1, 1, 0, bias=False), nn.BatchNorm2d(last_chn),
                                  HardSwish(), SELayer(last_chn) if mode == 'small' else nn.Sequential())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        chn_out = 1024 if mode == 'small' else 1280
        chn_out = _make_divisible(chn_out * width_mult, 8)
        self.classifier = nn.Sequential(
            nn.Conv2d(last_chn, chn_out, kernel_size=1, stride=1, bias=True),
            HardSwish(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(chn_out, n_classes, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        """Compute network output."""
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x


def mbv3_predefined_converter(slot):
    """MobileNetV3 original network Slot converter."""
    return MobileInvertedConvV3(slot.chn_in, slot.chn_out, slot.stride, **slot.kwargs)


@register_constructor
class MobileNetV3PredefinedConstructor(DefaultSlotTraversalConstructor):
    """MobileNetV3 original network constructor."""

    def convert(self, slot):
        """Convert slot to module."""
        return mbv3_predefined_converter(slot)


@register_constructor
class MobileNetV3SearchConstructor(DefaultMixedOpConstructor):
    """MobileNetV3 mixed operator search space Constructor."""

    def __init__(self, *args, fix_first=True, add_zero_op=True, keep_config=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.fix_first = fix_first
        self.add_zero_op = add_zero_op
        self.keep_config = keep_config
        self.predefined_fn = mbv3_predefined_converter
        self.first = True

    def convert(self, slot):
        """Convert Slot to mixed operator."""
        if self.fix_first and self.first:
            ent = self.predefined_fn(slot)
            self.first = False
        else:
            cands = self.candidates[:]
            cand_args = self.candidate_args.copy()
            if self.add_zero_op and slot.stride == 1 and slot.chn_in == slot.chn_out:
                self.candidates.append('NIL')
            if self.keep_config:
                self.candidate_args['use_hs'] = slot.kwargs['use_hs']
                self.candidate_args['use_se'] = slot.kwargs['use_se']
            ent = super().convert(slot)
            self.candidates = cands
            self.candidate_args = cand_args
        return ent


@register_constructor
class MobileNetV3ArchDescConstructor(DefaultSlotArchDescConstructor):
    """MobileNetV2 archdesc Constructor."""

    def __init__(self, *args, fix_first=True, keep_config=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.fix_first = fix_first
        self.keep_config = keep_config
        self.predefined_fn = mbv3_predefined_converter
        self.first = True

    def convert(self, slot):
        """Convert Slot to module from archdesc."""
        if self.fix_first and self.first:
            ent = self.predefined_fn(slot)
            self.first = False
            self.get_next_desc()
        else:
            fn_args = self.fn_args.copy()
            if self.keep_config:
                self.fn_args['use_hs'] = slot.kwargs['use_hs']
                self.fn_args['use_se'] = slot.kwargs['use_se']
            ent = super().convert(slot)
            self.fn_args = fn_args
        return ent


@register
def mobilenetv3_large(cfgs=None, **kwargs):
    """Construct a MobileNetV3-Large model."""
    cfgs = [
        # k, t, c, SE, NL, s
        [3, 0, 16, 0, 1, 2],
        [3, 16, 16, 0, 0, 1],
        [3, 64, 24, 0, 0, 2],
        [3, 72, 24, 0, 0, 1],
        [5, 72, 40, 1, 0, 2],
        [5, 120, 40, 1, 0, 1],
        [5, 120, 40, 1, 0, 1],
        [3, 240, 80, 0, 1, 2],
        [3, 200, 80, 0, 1, 1],
        [3, 184, 80, 0, 1, 1],
        [3, 184, 80, 0, 1, 1],
        [3, 480, 112, 1, 1, 1],
        [3, 672, 112, 1, 1, 1],
        [5, 672, 160, 1, 1, 2],
        [5, 960, 160, 1, 1, 1],
        [5, 960, 160, 1, 1, 1]
    ] if cfgs is None else cfgs
    return MobileNetV3(cfgs, mode='large', **kwargs)


@register
def mobilenetv3_small(cfgs=None, **kwargs):
    """Construct a MobileNetV3-Small model."""
    cfgs = [
        # k, t, c, SE, NL, s
        [3, 0, 16, 0, 1, 2],
        [3, 16, 16, 1, 0, 2],
        [3, 72, 24, 0, 0, 2],
        [3, 88, 24, 0, 0, 1],
        [5, 96, 40, 1, 1, 2],
        [5, 240, 40, 1, 1, 1],
        [5, 240, 40, 1, 1, 1],
        [5, 120, 48, 1, 1, 1],
        [5, 144, 48, 1, 1, 1],
        [5, 288, 96, 1, 1, 2],
        [5, 576, 96, 1, 1, 1],
        [5, 576, 96, 1, 1, 1],
    ] if cfgs is None else cfgs

    return MobileNetV3(cfgs, mode='small', **kwargs)
