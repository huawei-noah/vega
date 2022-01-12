# -*- coding:utf-8 -*-

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

"""Model weight initializer."""
import copy
import math
import torch.nn.init as init
from modnas.registry.construct import register


def _t_init_he_normal_fout(t, gain, fan_in, fan_out):
    stdv = gain / math.sqrt(fan_out)
    init.normal_(t, 0, stdv)


def _t_init_he_normal_fin(t, gain, fan_in, fan_out):
    stdv = gain / math.sqrt(fan_in)
    init.normal_(t, 0, stdv)


def _t_init_he_uniform_fout(t, gain, fan_in, fan_out):
    b = math.sqrt(3.) * gain / math.sqrt(fan_out)
    init.uniform_(t, -b, b)


def _t_init_he_uniform_fin(t, gain, fan_in, fan_out):
    b = math.sqrt(3.) * gain / math.sqrt(fan_in)
    init.uniform_(t, -b, b)


def _t_init_xavier_uniform(t, gain, fan_in, fan_out):
    b = math.sqrt(6.) * gain / math.sqrt(fan_in + fan_out)
    init.uniform_(t, -b, b)


def _t_init_xavier_normal(t, gain, fan_in, fan_out):
    stdv = math.sqrt(2.) * gain / math.sqrt(fan_in + fan_out)
    init.normal_(t, 0, stdv)


def _t_init_uniform_fin(t, gain, fan_in, fan_out):
    b = 1.0 / math.sqrt(fan_in)
    init.uniform_(t, -b, b)


def _t_init_uniform_fout(t, gain, fan_in, fan_out):
    b = 1.0 / math.sqrt(fan_out)
    init.uniform_(t, -b, b)


def _t_init_uniform(t, gain, fan_in, fan_out):
    init.uniform_(t)


def _t_init_normal(t, gain, fan_in, fan_out):
    init.normal_(t)


def _t_init_zeros(t, gain, fan_in, fan_out):
    init.zeros_(t)


def _t_init_ones(t, gain, fan_in, fan_out):
    init.ones_(t)


def _init_tensor(init_type, t, gain, fan_in, fan_out):
    init_fn = _tensor_init_fn.get(init_type)
    if init_fn is None or t is None:
        return
    init_fn(t, gain, fan_in, fan_out)


def _m_init_conv(m, config):
    init_type = config['conv']['type']
    bias_init_type = config['bias']['type']
    gain = config['gain']
    if init_type is None:
        return
    rec_size = m.kernel_size[0] * m.kernel_size[1]
    fan_in = rec_size * m.in_channels
    fan_out = rec_size * m.out_channels
    if config['conv'].get('div_groups', True):
        fan_in /= m.groups
        fan_out /= m.groups
    _init_tensor(init_type, m.weight, gain, fan_in, fan_out)
    if m.bias is not None:
        _init_tensor(bias_init_type, m.bias, gain, fan_in, fan_out)


def _m_init_norm(m, config):
    init_type = config['norm']['type']
    bias_init_type = config['bias']['type']
    momentum = config['norm'].get('momentum')
    eps = config['norm'].get('eps')
    gain = config['gain']
    m.reset_running_stats()
    if momentum is not None:
        m.momentum = momentum
    if eps is not None:
        m.eps = eps
    if not m.affine:
        return
    fan_in = fan_out = m.num_features
    _init_tensor(init_type, m.weight, gain, fan_in, fan_out)
    _init_tensor(bias_init_type, m.bias, gain, fan_in, fan_out)


def _m_init_fc(m, config):
    init_type = config['fc']['type']
    bias_init_type = config['bias']['type']
    gain = config['gain']
    if init_type is None:
        return
    fan_in, fan_out = m.in_features, m.out_features
    _init_tensor(init_type, m.weight, gain, fan_in, fan_out)
    if m.bias is None:
        return
    _init_tensor(bias_init_type, m.bias, gain, fan_in, fan_out)


_tensor_init_fn = {k[8:]: v for (k, v) in globals().items() if k.startswith('_t_init_')}
_module_init_fn = {k[8:]: v for (k, v) in globals().items() if k.startswith('_m_init_')}


_default_init_config = {
    'conv': {
        'type': None,
        'div_groups': True,
    },
    'norm': {
        'type': None,
    },
    'fc': {
        'type': None,
    },
    'bias': {
        'type': None,
    },
}


_default_module_map = {
    'Conv2d': 'conv',
    'BatchNorm2d': 'norm',
    'GroupNorm': 'norm',
    'Linear': 'fc',
}


@register
class DefaultModelInitializer():
    """Model weight initializer class."""

    def __init__(self,
                 init_config=None,
                 module_init_map=None,
                 default_init_type=None,
                 neg_slope=math.sqrt(5),
                 nonlinear='leaky_relu'):
        self.init_config = copy.deepcopy(_default_init_config)
        self.init_config['gain'] = init.calculate_gain(nonlinear, neg_slope)
        self.init_config.update(init_config or {})
        self.module_init_map = _default_module_map.copy()
        self.module_init_map.update(module_init_map or {})
        self.default_init_type = default_init_type

    def __call__(self, model):
        """Return initialized model."""
        for m in model.modules():
            m_init_type = self.module_init_map.get(type(m).__name__)
            if m_init_type is not None:
                _module_init_fn[m_init_type](m, self.init_config)
            elif len(list(m.children())) == 0:
                for p in m.parameters():
                    sz = p.shape
                    fan_out = sz[0] if len(sz) else 1
                    fan_in = sz[min(1, len(sz) - 1)] if len(sz) else 1
                    _init_tensor(self.default_init_type, p, self.init_config['gain'], fan_in, fan_out)
        return model
