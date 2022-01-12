# -*- coding: utf-8 -*-

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

"""Prune ArchSpace."""

import logging
import vega
from vega import is_torch_backend, is_tf_backend
from vega.modules.operators import ops
from vega.common.class_factory import ClassFactory
from vega.modules.arch.architecture import Architecture


def _to_cpu(data):
    try:
        import torch
        if torch.is_tensor(data):
            return data.cpu()
    except Exception:
        logging.debug('Falied to convert data to cpu.')

    if isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple([_to_cpu(v) for v in data])
    return data


@ClassFactory.register('Prune', 'Conv2d')
class Conv2dPruneArchitecture(Architecture):
    """Prune Conv2d."""

    @staticmethod
    def decode(value, org_value):
        """Decode arch desc."""
        return sum(value)

    @staticmethod
    def fit_weights(module, x):
        """Fit weight."""
        arch_params = module.module_arch_params
        if not arch_params:
            return None
        freeze(module)
        prune_conv2d_out_channels(arch_params, module)
        prune_conv2d_in_channels(arch_params, module)
        return None


@ClassFactory.register('Prune', 'BatchNorm2d')
class BatchNorm2dPruneArchitecture(Architecture):
    """Prune BatchNorm."""

    @staticmethod
    def decode(value, org_value):
        """Decode arch desc."""
        return sum(value)

    @staticmethod
    def fit_weights(module, x):
        """Fit weights shape."""
        arch_params = module.module_arch_params
        if not arch_params:
            return None
        adapt(module)
        idx = [idx for idx, value in enumerate(arch_params.num_features) if value == 1]
        weights = module.get_weights()
        weights = _to_cpu(weights)
        for name, weight in weights.items():
            if name in ["total_ops", "total_params"]:
                continue
            if is_tf_backend():
                module.set_weights(name, weight[idx])
            else:
                module.set_weights(name, weight[idx].to(vega.get_devices()))
        return None


@ClassFactory.register('Prune', 'Linear')
class LinearPruneArchitecture(Architecture):
    """Prune."""

    @staticmethod
    def decode(value, org_value):
        """Decode arch params."""
        return sum(value)

    @staticmethod
    def fit_weights(module, x):
        """Fit weights shape."""
        arch_params = module.module_arch_params
        if not arch_params:
            return None
        idx_in = [idx for idx, value in enumerate(arch_params.in_features) if value == 1]
        weights = module.get_weights()
        for name, weight in weights.items():
            if name in ["total_ops", "total_params"]:
                continue
            if 'kernel' in name or 'weight' in name:
                if is_tf_backend():
                    module.set_weights(name, weight[idx_in, :])
                elif is_torch_backend():
                    module.set_weights(name, weight[:, idx_in].to(vega.get_devices()))
                else:
                    module.set_weights(name, weight[idx_in, :].to(vega.get_devices()))
        return None


@ClassFactory.register('Prune', 'Reshape')
class ReshapePruneArchitecture(Architecture):
    """Prune Reshape ops."""

    @staticmethod
    def decode(value, org_value):
        """Decode arch params."""
        return [org_value[0], sum(value)]

    @staticmethod
    def fit_weights(module, x):
        """Do nothing."""
        return None


def freeze(module):
    """Freeze parameter."""
    if not is_torch_backend():
        return
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(False)


def adapt(module):
    """Adapt mean and var in dataset."""
    if not is_torch_backend():
        return
    module.weight.requires_grad = False
    module.bias.requires_grad = False


def prune_conv2d_out_channels(arch_params, module):
    """Prune out channels of conv2d."""
    weights = module.get_weights()
    weights = _to_cpu(weights)
    if arch_params.get('out_channels'):
        out_channels_idx = [idx for idx, value in enumerate(arch_params.out_channels) if value == 1]
        for name, weight in weights.items():
            if weight is None:
                continue
            if name in ["total_ops", "total_params"]:
                continue
            if 'BatchNorm' in name:
                if is_tf_backend():
                    module.set_weights(name, weight[out_channels_idx])
                else:
                    module.set_weights(name, weight[out_channels_idx].to(vega.get_devices()))
            else:
                if is_tf_backend():
                    module.set_weights(name, weight[:, :, :, out_channels_idx])
                elif is_torch_backend():
                    module.set_weights(name, weight[out_channels_idx, :, :, :].to(vega.get_devices()))
                else:
                    module.set_weights(name, weight[:, :, :, out_channels_idx].to(vega.get_devices()))


def prune_conv2d_in_channels(arch_params, module):
    """Prune in channels of conv2d."""
    weights = module.get_weights()
    weights = _to_cpu(weights)
    in_channels = module.in_channels
    out_channels = module.out_channels
    if arch_params.get('in_channels'):
        in_channels_idx = [idx for idx, value in enumerate(arch_params.in_channels) if value == 1]
        for name, weight in weights.items():
            if name in ["total_ops", "total_params"]:
                continue
            if weight is None or 'BatchNorm' in name:
                continue
            if weight is not None:
                if is_torch_backend():
                    if module.groups == 1:
                        module.set_weights(name, weight[:, in_channels_idx, :, :].to(vega.get_devices()))
                    else:
                        module.groups = min(in_channels, out_channels)
                    if module.groups < in_channels:
                        in_channels_diff = int(in_channels) - int(weight.shape[1] * module.groups)
                        in_channels_group_diff = int(in_channels_diff / module.groups)
                        padding = [0, 0, 0, 0, 0, 0, 0, 0]
                        padding[5] = in_channels_group_diff
                        module.set_weights(name, ops.pad(weight, padding))
                else:
                    if is_tf_backend():
                        module.set_weights(name, weight[:, :, in_channels_idx, :])
                    else:
                        module.set_weights(name, weight[:, :, in_channels_idx, :].to(vega.get_devices()))
