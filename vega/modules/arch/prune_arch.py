# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Prune ArchSpace."""
from vega import is_torch_backend
from vega.common.class_factory import ClassFactory
from vega.modules.arch.architecture import Architecture


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
        weights = module.get_weights()
        if arch_params.get('out_channels'):
            out_channels_idx = [idx for idx, value in enumerate(arch_params.out_channels) if value == 1]
            for name, weight in weights.items():
                if weight is None:
                    continue
                if 'BatchNorm' in name:
                    module.set_weights(name, weight[out_channels_idx])
                else:
                    if is_torch_backend():
                        module.set_weights(name, weight[out_channels_idx, :, :, :])
                    else:
                        module.set_weights(name, weight[:, :, :, out_channels_idx])
        if arch_params.get('in_channels'):
            in_channels_idx = [idx for idx, value in enumerate(arch_params.in_channels) if value == 1]
            for name, weight in weights.items():
                if weight is None or 'BatchNorm' in name:
                    continue
                if weight is not None:
                    if is_torch_backend():
                        module.set_weights(name, weight[:, in_channels_idx, :, :])
                    else:
                        module.set_weights(name, weight[:, :, in_channels_idx, :])
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
        idx = [idx for idx, value in enumerate(arch_params.num_features) if value == 1]
        weights = module.get_weights()
        for name, weight in weights.items():
            module.set_weights(name, weight[idx])
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
            if 'kernel' in name or 'weight' in name:
                if is_torch_backend():
                    module.set_weights(name, weight[:, idx_in])
                else:
                    module.set_weights(name, weight[idx_in, :])
        return None
