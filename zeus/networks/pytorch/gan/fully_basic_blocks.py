
# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined ResNet Blocks For Detection."""
from torch import nn
import torch.nn.functional as F


# 7
PRIMITIVES = [
    'none',
    'skip_connect',
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# 3
PRIMITIVES_up = [
    'nearest',
    'bilinear',
    'ConvTranspose'
]

# 6
PRIMITIVES_down = [
    'avg_pool',
    'max_pool',
    'conv_3x3',
    'conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


# ------------------------------------------------------------------------------------------------------------------- #

OPS = {
    'none': lambda in_ch, out_ch, stride, sn, act: Zero(),
    'skip_connect': lambda in_ch, out_ch, stride, sn, act: Identity(),
    'conv_1x1': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 1, stride, 0, sn, act),
    'conv_3x3': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 3, stride, 1, sn, act),
    'conv_5x5': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 5, stride, 2, sn, act),
    'dil_conv_3x3': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 3, stride, 2, 2, sn, act),
    'dil_conv_5x5': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 5, stride, 4, 2, sn, act)
}

OPS_down = {
    'avg_pool': lambda in_ch, out_ch, stride, sn, act: Pool(in_ch, out_ch, mode='Avg'),
    'max_pool': lambda in_ch, out_ch, stride, sn, act: Pool(in_ch, out_ch, mode='Max'),
    'conv_3x3': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 3, stride, 1, sn, act),
    'conv_5x5': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 5, stride, 2, sn, act),
    'dil_conv_3x3': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 3, stride, 2, 2, sn, act),
    'dil_conv_5x5': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 5, stride, 4, 2, sn, act)
}

UPS = {
    'nearest': lambda in_ch, out_ch: Up(in_ch, out_ch, mode='nearest'),
    'bilinear': lambda in_ch, out_ch: Up(in_ch, out_ch, mode='bilinear'),
    'ConvTranspose': lambda in_ch, out_ch: Up(in_ch, out_ch, mode='convT')
}

# ------------------------------------------------------------------------------------------------------------------- #


class Conv(nn.Module):
    """Conv class."""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, sn, act):
        super(Conv, self).__init__()
        if sn:
            self.conv = nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding))
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                                  stride=stride, padding=padding)
        if act:
            self.op = nn.Sequential(nn.ReLU(), self.conv)
        else:
            self.op = nn.Sequential(self.conv)

    def forward(self, x):
        """call."""
        return self.op(x)


class DilConv(nn.Module):
    """DilConv class."""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, sn, act):
        super(DilConv, self).__init__()
        if sn:
            self.dilconv = nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
        else:
            self.dilconv = \
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                          stride=stride, padding=padding, dilation=dilation)
        if act:
            self.op = nn.Sequential(nn.ReLU(), self.dilconv)
        else:
            self.op = nn.Sequential(self.dilconv)

    def forward(self, x):
        """call."""
        return self.op(x)


class Identity(nn.Module):
    """Identity class."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """call."""
        return x


class Zero(nn.Module):
    """Zero class."""

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        """call."""
        return x.mul(0.)


class Up(nn.Module):
    """Up class."""

    def __init__(self, in_ch, out_ch, mode=None):
        super(Up, self).__init__()
        self.up_mode = mode
        if self.up_mode == 'convT':
            self.convT = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(
                    in_ch, in_ch, kernel_size=3, stride=2, padding=1, output_padding=1, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)
            )
        else:
            self.c = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
            )

    def forward(self, x):
        """call."""
        if self.up_mode == 'convT':
            return self.convT(x)
        else:
            return self.c(F.interpolate(x, scale_factor=2, mode=self.up_mode))


class Pool(nn.Module):
    """Pool class."""

    def __init__(self, in_ch, out_ch, mode=None):
        super(Pool, self).__init__()
        if mode == 'Avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        elif mode == 'Max':
            self.pool = nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1)

    def forward(self, x):
        """call."""
        return self.pool(x)


class MixedOp(nn.Module):
    """MixedOp class."""

    def __init__(self, in_ch, out_ch, stride, sn, act, primitives):
        super(MixedOp, self).__init__()
        self.ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](in_ch, out_ch, stride, sn, act)
            self.ops.append(op)

    def forward(self, x):
        """call."""
        return sum(op(x) for op in self.ops)


class MixedUp(nn.Module):
    """MixedUp class."""

    def __init__(self, in_ch, out_ch, primitives):
        super(MixedUp, self).__init__()
        self.ups = nn.ModuleList()
        for primitive in primitives:
            up = UPS[primitive](in_ch, out_ch)
            self.ups.append(up)

    def forward(self, x):
        """call."""
        return sum(up(x) for up in self.ups)


class MixedDown(nn.Module):
    """MixedDown class."""

    def __init__(self, in_ch, out_ch, stride, sn, act, primitives):
        super(MixedDown, self).__init__()
        self.ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS_down[primitive](in_ch, out_ch, stride, sn, act)
            self.ops.append(op)

    def forward(self, x):
        """call."""
        return sum(op(x) for op in self.ops)


class Cell(nn.Module):
    """Cell class."""

    def __init__(self, in_channels, out_channels, up_mode, genotype, num_skip_in=0, norm=None):
        super(Cell, self).__init__()

        self.up0 = MixedUp(in_channels, out_channels, [
                           PRIMITIVES_up[genotype[0]]])
        self.up1 = MixedUp(in_channels, out_channels, [
                           PRIMITIVES_up[genotype[1]]])
        if genotype[2] > 0:
            self.c0 = MixedOp(out_channels, out_channels, 1,
                              False, True, [PRIMITIVES[genotype[2]]])
        if genotype[3] > 0:
            self.c1 = MixedOp(out_channels, out_channels, 1,
                              False, True, [PRIMITIVES[genotype[3]]])
        if genotype[4] > 0:
            self.c2 = MixedOp(out_channels, out_channels, 1,
                              False, True, [PRIMITIVES[genotype[4]]])
        if genotype[5] > 0:
            self.c3 = MixedOp(out_channels, out_channels, 1,
                              False, True, [PRIMITIVES[genotype[5]]])
        if genotype[6] > 0:
            self.c4 = MixedOp(out_channels, out_channels, 1,
                              False, True, [PRIMITIVES[genotype[6]]])

        self.up_mode = up_mode
        self.norm = norm

        # no norm
        if norm:
            if norm == 'bn':
                self.n1 = nn.BatchNorm2d(in_channels)
                self.n2 = nn.BatchNorm2d(out_channels)
            elif norm == 'in':
                self.n1 = nn.InstanceNorm2d(in_channels)
                self.n2 = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError(norm)

        # cross scale skip
        self.skip_in_ops = None
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=1)
                 for _ in range(num_skip_in)]
            )

    def forward(self, x, skip_ft=None):
        """call."""
        node0 = self.up0(x)
        node1 = self.up1(x)
        _, _, ht, wt = node0.size()

        # for different topologies
        if hasattr(self, 'c0'):
            node2 = self.c0(node0)
            if hasattr(self, 'c1'):
                node2 = node2 + self.c1(node1)
        else:
            node2 = self.c1(node1)

        # skip out feat
        h_skip_out = node2

        # skip in feat
        if self.skip_in_ops:
            assert len(self.skip_in_ops) == len(skip_ft)
            for ft, skip_in_op in zip(skip_ft, self.skip_in_ops):
                node2 += skip_in_op(F.interpolate(ft,
                                                  size=(ht, wt), mode=self.up_mode))

        # for different topologies
        if hasattr(self, 'c2'):
            node3 = self.c2(node0)
            if hasattr(self, 'c3'):
                node3 = node3 + self.c3(node1)
                if hasattr(self, 'c4'):
                    node3 = node3 + self.c4(node2)
            else:
                if hasattr(self, 'c4'):
                    node3 = node3 + self.c4(node2)
        else:
            if hasattr(self, 'c3'):
                node3 = self.c3(node1)
                if hasattr(self, 'c4'):
                    node3 = node3 + self.c4(node2)
            else:
                node3 = self.c4(node2)

        return h_skip_out, node3


def _downsample(x):
    """Downsample with Avg Pooling."""
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    """OptimizedDisBlock class."""

    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation
        self.c1 = nn.Conv2d(in_channels, out_channels,
                            kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels,
                            kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, padding=0)
        self.c1 = nn.utils.spectral_norm(self.c1)
        self.c2 = nn.utils.spectral_norm(self.c2)
        self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        """call."""
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        """call."""
        return self.c_sc(_downsample(x))

    def forward(self, x):
        """call."""
        return self.residual(x) + self.shortcut(x)


class DisCell(nn.Module):
    """DisCell class."""

    def __init__(self, in_channels, out_channels, hidden_channels=None, activation=nn.ReLU(), genotype=None):
        super(DisCell, self).__init__()
        if genotype[5] >= 0:
            self.down0 = MixedDown(in_channels, out_channels, 2, True, True, [
                                   PRIMITIVES_down[genotype[5]]])
            self.down1 = MixedDown(in_channels, out_channels, 2, True, True, [
                                   PRIMITIVES_down[genotype[6]]])
        if genotype[0] > 0:
            self.c0 = MixedOp(out_channels, out_channels, 1,
                              True, True, [PRIMITIVES[genotype[0]]])
        if genotype[1] > 0:
            self.c1 = MixedOp(out_channels, out_channels, 1,
                              True, True, [PRIMITIVES[genotype[1]]])
        if genotype[2] > 0:
            self.c2 = MixedOp(out_channels, out_channels, 1,
                              True, True, [PRIMITIVES[genotype[2]]])
        if genotype[3] > 0:
            self.c3 = MixedOp(out_channels, out_channels, 1,
                              True, True, [PRIMITIVES[genotype[3]]])
        if genotype[4] > 0:
            self.c4 = MixedOp(out_channels, out_channels, 1,
                              True, False, [PRIMITIVES[genotype[4]]])

    def forward(self, x):
        """call."""
        node0 = x
        node1 = self.c0(node0)
        if hasattr(self, 'c1'):
            node2 = self.c1(node0)
            if hasattr(self, 'c2'):
                node2 = node2 + self.c2(node1)
        else:
            node2 = self.c2(node1)
        if hasattr(self, 'c3'):
            node3 = self.c3(node1)
            if hasattr(self, 'c4'):
                node3 = node3 + self.c4(node0)
        else:
            node3 = self.c4(node0)
        if hasattr(self, 'down0'):
            node4 = self.down0(node2) + self.down1(node3)
        else:
            node4 = node2 + node3
        return node4
