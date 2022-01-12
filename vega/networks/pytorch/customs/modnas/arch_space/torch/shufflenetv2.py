# -*- coding:utf-8 -*-

# This file is adapted from the SinglePathOneShot library at
# https://github.com/megvii-model/SinglePathOneShot

# 2020.6.29-Changed for Modular-NAS search space.
#         Huawei Technologies Co., Ltd. <linyunfeng5@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.

"""ShuffleNetV2 architectures."""

import torch
import torch.nn as nn
from modnas.registry.construct import register as register_constructor
from modnas.registry.construct import DefaultMixedOpConstructor, DefaultSlotTraversalConstructor
from modnas.registry.arch_space import build, register
from ..slot import register_slot_ccs
from .. import ops
from ..slot import Slot


kernel_sizes = [3, 5, 7, 9]
for k in kernel_sizes:
    register_slot_ccs(
        lambda C_in, C_out, S, chn_mid=None, ks=k: ShuffleUnit(C_in, C_out, S, ksize=ks, chn_mid=chn_mid),
        'SHU{}'.format(k))
    register_slot_ccs(
        lambda C_in, C_out, S, chn_mid=None, ks=k: ShuffleUnitXception(C_in, C_out, S, ksize=ks, chn_mid=chn_mid),
        'SHX{}'.format(k))


def channel_split(x, split):
    """Return data split in channel dimension."""
    if x.size(1) == split * 2:
        return torch.split(x, split, dim=1)
    else:
        raise ValueError('Failed to return data split in channel dimension.')


def shuffle_channels(x, groups=2):
    """Return data shuffled in channel dimension."""
    batch_size, channels, height, width = x.size()
    if channels % groups == 0:
        channels_per_group = channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, channels, height, width)
        return x
    else:
        raise ValueError('Failed to return data shuffled in channel dimension.')


class ShuffleUnit(nn.Module):
    """ShuffleNetV2 unit class."""

    def __init__(self, chn_in, chn_out, stride, ksize, chn_mid=None):
        super(ShuffleUnit, self).__init__()
        chn_in = chn_in // 2 if stride == 1 else chn_in
        chn_mid = int(chn_out // 2) if chn_mid is None else chn_mid
        self.stride = stride
        self.ksize = ksize
        self.chn_in = chn_in
        pad = ksize // 2

        outputs = chn_out - chn_in

        branch_main = [
            # pw
            nn.Conv2d(chn_in, chn_mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chn_mid, **ops.config.bn),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(chn_mid, chn_mid, ksize, stride, pad, groups=chn_mid, bias=False),
            nn.BatchNorm2d(chn_mid, **ops.config.bn),
            # pw-linear
            nn.Conv2d(chn_mid, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs, **ops.config.bn),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(chn_in, chn_in, ksize, stride, pad, groups=chn_in, bias=False),
                nn.BatchNorm2d(chn_in, **ops.config.bn),
                # pw-linear
                nn.Conv2d(chn_in, chn_in, 1, 1, 0, bias=False),
                nn.BatchNorm2d(chn_in, **ops.config.bn),
                nn.ReLU(inplace=True),
            ]
        else:
            branch_proj = []
        self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, x):
        """Return network output."""
        if self.stride == 1:
            x_proj, x = channel_split(x, self.chn_in)
        elif self.stride == 2:
            x_proj = x
        x = torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)
        x = shuffle_channels(x)
        return x


class ShuffleUnitXception(nn.Module):
    """ShuffleNetV2 Xception unit class."""

    def __init__(self, chn_in, chn_out, stride, ksize=3, chn_mid=None):
        super(ShuffleUnitXception, self).__init__()
        chn_in = chn_in // 2 if stride == 1 else chn_in
        chn_mid = int(chn_out // 2) if chn_mid is None else chn_mid
        self.stride = stride
        self.ksize = ksize
        self.chn_in = chn_in
        outputs = chn_out - chn_in
        pad = ksize // 2

        branch_main = [
            # dw
            nn.Conv2d(chn_in, chn_in, ksize, stride, pad, groups=chn_in, bias=False),
            nn.BatchNorm2d(chn_in, **ops.config.bn),
            # pw
            nn.Conv2d(chn_in, chn_mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chn_mid, **ops.config.bn),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(chn_mid, chn_mid, ksize, 1, pad, groups=chn_mid, bias=False),
            nn.BatchNorm2d(chn_mid, **ops.config.bn),
            # pw
            nn.Conv2d(chn_mid, chn_mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chn_mid, **ops.config.bn),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(chn_mid, chn_mid, ksize, 1, pad, groups=chn_mid, bias=False),
            nn.BatchNorm2d(chn_mid, **ops.config.bn),
            # pw
            nn.Conv2d(chn_mid, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs, **ops.config.bn),
            nn.ReLU(inplace=True),
        ]

        self.branch_main = nn.Sequential(*branch_main)

        if self.stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(chn_in, chn_in, ksize, stride, pad, groups=chn_in, bias=False),
                nn.BatchNorm2d(chn_in, **ops.config.bn),
                # pw-linear
                nn.Conv2d(chn_in, chn_in, 1, 1, 0, bias=False),
                nn.BatchNorm2d(chn_in, **ops.config.bn),
                nn.ReLU(inplace=True),
            ]
        else:
            branch_proj = []
        self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, x):
        """Return network output."""
        if self.stride == 1:
            x_proj, x = channel_split(x, self.chn_in)
        elif self.stride == 2:
            x_proj = x
        x = torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)
        x = shuffle_channels(x)
        return x


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2 class."""

    def __init__(self, cfgs, chn_in=3, n_classes=1000, dropout_rate=0.1):
        super(ShuffleNetV2, self).__init__()
        self.out_channels = [cfg[0] for cfg in cfgs]
        self.num_repeats = [cfg[1] for cfg in cfgs]
        self.strides = [cfg[2] for cfg in cfgs]
        self.expansions = [cfg[3] for cfg in cfgs]
        features = []
        for i, (c, n, s, e) in enumerate(cfgs):
            if i == 0:
                features.append(self._get_stem(chn_in, c, s))
            elif i == len(cfgs) - 1:
                features.append(
                    nn.Sequential(
                        nn.Conv2d(chn_in, c, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(c, affine=True),
                        nn.ReLU(inplace=True),
                    ))
            else:
                for j in range(n):
                    block_stride = s if j == 0 else 1
                    chn_mid = int(c // 2 * e)
                    features.append(Slot(_chn_in=chn_in, _chn_out=c, _stride=block_stride, chn_mid=chn_mid))
                    chn_in = c
            chn_in = c
        self.features = nn.Sequential(*features)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(chn_in, n_classes, bias=False)
        self._initialize_weights()

    def _get_stem(self, chn_in, chn, stride=2):
        """Return stem layers."""
        if stride == 4:
            return nn.Sequential(
                nn.Conv2d(chn_in, chn, 3, 2, 1, bias=False),
                nn.BatchNorm2d(chn, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
            )
        return nn.Sequential(
            nn.Conv2d(chn_in, chn, 3, stride, 1, bias=False),
            nn.BatchNorm2d(chn, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Return network output."""
        x = self.features(x)
        x = self.globalpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize weights for all modules."""
        first_conv = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if first_conv:
                    nn.init.normal_(m.weight, 0, 0.01)
                    first_conv = False
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


@register_constructor
class ShuffleNetV2SearchConstructor(DefaultMixedOpConstructor):
    """ShuffleNetV2 mixed operator search space constructor."""

    def __init__(self, *args, add_identity_op=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_identity_op = add_identity_op

    def convert(self, slot):
        """Convert slot to mixed operator."""
        cands = self.candidates[:]
        if self.add_identity_op and slot.stride == 1 and slot.chn_in == slot.chn_out:
            self.candidates.append('IDT')
        ent = super().convert(slot)
        self.candidates = cands
        return ent


@register_constructor
class ShuffleNetV2PredefinedConstructor(DefaultSlotTraversalConstructor):
    """ShuffleNetV2 original network constructor."""

    def convert(self, slot):
        """Convert slot to module."""
        return build('SHU3', slot)


@register
def shufflenetv2_oneshot(cfgs=None, **kwargs):
    """Return a ShuffleNetV2 oneshot model."""
    cfgs = [
        [16, 1, 2, 1.0],
        [64, 4, 2, 1.0],
        [160, 4, 2, 1.0],
        [320, 8, 2, 1.0],
        [640, 4, 2, 1.0],
        [1024, 1, 1, 1.0],
    ] if cfgs is None else cfgs
    return ShuffleNetV2(cfgs=cfgs, **kwargs)


@register
def cifar_shufflenetv2_oneshot(cfgs=None, **kwargs):
    """Return a ShuffleNetV2 oneshot model for CIFAR dataset."""
    cfgs = [
        [24, 1, 1, 1.0],
        [64, 4, 2, 1.0],
        [160, 4, 2, 1.0],
        [320, 8, 2, 1.0],
        [640, 4, 1, 1.0],
        [1024, 1, 1, 1.0],
    ] if cfgs is None else cfgs
    return ShuffleNetV2(cfgs=cfgs, **kwargs)


@register
def shufflenetv2(cfgs=None, **kwargs):
    """Return a ShuffleNetV2 model."""
    cfgs = [
        [24, 1, 4, 1.0],
        [116, 4, 2, 1.0],
        [232, 8, 2, 1.0],
        [464, 4, 2, 1.0],
        [1024, 1, 1, 1.0],
    ] if cfgs is None else cfgs
    return ShuffleNetV2(cfgs=cfgs, **kwargs)


@register
def cifar_shufflenetv2(cfgs=None, **kwargs):
    """Return a ShuffleNetV2 model for CIFAR dataset."""
    cfgs = [
        [24, 1, 1, 1.0],
        [116, 4, 2, 1.0],
        [232, 8, 2, 1.0],
        [464, 4, 2, 1.0],
        [1024, 1, 1, 1.0],
    ] if cfgs is None else cfgs
    return ShuffleNetV2(cfgs=cfgs, **kwargs)
