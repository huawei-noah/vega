# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Model weight initializer."""
import math
import torch.nn as nn
from modnas.registry.construct import register


def _init_he_normal_fout(t, gain, fan_in, fan_out):
    stdv = gain / math.sqrt(fan_out)
    nn.init.normal_(t, 0, stdv)


def _init_he_normal_fin(t, gain, fan_in, fan_out):
    stdv = gain / math.sqrt(fan_in)
    nn.init.normal_(t, 0, stdv)


def _init_he_uniform_fout(t, gain, fan_in, fan_out):
    b = math.sqrt(3.) * gain / math.sqrt(fan_out)
    nn.init.uniform_(t, -b, b)


def _init_he_uniform_fin(t, gain, fan_in, fan_out):
    b = math.sqrt(3.) * gain / math.sqrt(fan_in)
    nn.init.uniform_(t, -b, b)


def _init_xavier_uniform(t, gain, fan_in, fan_out):
    b = math.sqrt(6.) * gain / math.sqrt(fan_in + fan_out)
    nn.init.uniform_(t, -b, b)


def _init_xavier_normal(t, gain, fan_in, fan_out):
    stdv = math.sqrt(2.) * gain / math.sqrt(fan_in + fan_out)
    nn.init.normal_(t, 0, stdv)


def _init_uniform_fin(t, gain, fan_in, fan_out):
    b = 1.0 / math.sqrt(fan_in)
    nn.init.uniform_(t, -b, b)


def _init_uniform_fout(t, gain, fan_in, fan_out):
    b = 1.0 / math.sqrt(fan_out)
    nn.init.uniform_(t, -b, b)


def _init_uniform(t, gain, fan_in, fan_out):
    nn.init.uniform_(t)


def _init_normal(t, gain, fan_in, fan_out):
    nn.init.normal_(t)


def _init_zeros(t, gain, fan_in, fan_out):
    nn.init.zeros_(t)


def _init_ones(t, gain, fan_in, fan_out):
    nn.init.ones_(t)


_initializers = {k[5:]: v for (k, v) in globals().items() if k.startswith('_init_')}


@register
class DefaultModelInitializer():
    """Model weight initializer class."""

    def __init__(self,
                 default_init_type=None,
                 conv_init_type=None,
                 conv_div_groups=True,
                 bn_init_type=None,
                 bn_momentum=None,
                 bn_eps=None,
                 fc_init_type=None,
                 bias_init_type=None,
                 neg_slope=math.sqrt(5),
                 nonlinear='leaky_relu'):
        self.default_init_type = default_init_type
        self.conv_init_type = conv_init_type
        self.conv_div_groups = conv_div_groups
        self.bn_init_type = bn_init_type
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.fc_init_type = fc_init_type
        self.bias_init_type = bias_init_type
        self.neg_slope = neg_slope
        self.nonlinear = nonlinear
        self.gain = nn.init.calculate_gain(nonlinear, neg_slope)

    def _init_tensor(self, init_type, t, gain, fan_in, fan_out):
        if init_type not in _initializers or t is None:
            return
        init_fn = _initializers[init_type]
        init_fn(t, gain, fan_in, fan_out)

    def __call__(self, model):
        """Return initialized model."""
        gain = self.gain
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if self.conv_init_type is None:
                    continue
                rec_size = m.kernel_size[0] * m.kernel_size[1]
                fan_in = rec_size * m.in_channels
                fan_out = rec_size * m.out_channels
                if self.conv_div_groups:
                    fan_in /= m.groups
                    fan_out /= m.groups
                self.init_tensor(self.conv_init_type, m.weight, gain, fan_in, fan_out)
                if m.bias is not None:
                    self.init_tensor(self.bias_init_type, m.bias, gain, fan_in, fan_out)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.reset_running_stats()
                if self.bn_momentum is not None:
                    m.momentum = self.bn_momentum
                if self.bn_eps is not None:
                    m.eps = self.bn_eps
                if not m.affine:
                    continue
                fan_in = fan_out = m.num_features
                self.init_tensor(self.bn_init_type, m.weight, gain, fan_in, fan_out)
                self.init_tensor(self.bias_init_type, m.bias, gain, fan_in, fan_out)
            elif isinstance(m, nn.Linear):
                if self.fc_init_type is None:
                    continue
                self.init_tensor(self.fc_init_type, m.weight, gain, fan_in, fan_out)
                if m.bias is None:
                    continue
                self.init_tensor(self.bias_init_type, m.bias, gain, fan_in, fan_out)
            elif len(list(m.children())) == 0:
                for p in m.parameters():
                    sz = p.shape
                    fan_out = sz[0] if len(sz) else 1
                    fan_in = sz[min(1, len(sz) - 1)] if len(sz) else 1
                    self.init_tensor(self.default_init_type, p, gain, fan_in, fan_out)
