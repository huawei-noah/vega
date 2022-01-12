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
"""Ops ArchSpace."""
from vega import is_torch_backend
from vega.common.class_factory import ClassFactory
from vega.modules.arch.architecture import Architecture
from vega.modules.operators import ops


@ClassFactory.register('double_channels', 'Conv2d')
class Conv2dDoubleChannelArchitecture(Architecture):
    """Double Channel."""

    @staticmethod
    def decode(value, org_value):
        """Decode arch params."""
        if value == 1:
            return org_value // 2
        elif value == 2:
            return org_value * 2
        else:
            return org_value

    @staticmethod
    def fit_weights(module, x):
        """Fit weight."""
        inputs = x[0] if isinstance(x, tuple) else x
        for name, weight in module.get_weights().items():
            if weight is None:
                continue
            in_channels_axis = 1 if is_torch_backend() else 2
            out_channels_axis = 0 if is_torch_backend() else 3
            if 'BatchNorm' in name:
                out_channels_diff = int(module.out_channels) - int(weight.shape[0])
                if out_channels_diff == 0:
                    continue
                padding = [0, out_channels_diff]
            else:
                groups = module.groups
                if groups == module.in_channels and module.out_channels < groups:
                    module.out_channels = groups
                in_channels_diff = int(inputs.shape[1]) - int(weight.shape[in_channels_axis] * module.groups)
                out_channels_diff = int(module.out_channels) - int(weight.shape[out_channels_axis])
                if in_channels_diff == 0 and out_channels_diff == 0:
                    continue
                padding = [0, 0, 0, 0, 0, 0, 0, 0]
                if groups == 1:
                    if in_channels_diff != 0:
                        padding[5] = in_channels_diff
                        module.in_channels += in_channels_diff
                else:
                    if in_channels_diff > 0:
                        in_channels_group_diff = int(in_channels_diff / groups)
                        padding[5] = in_channels_group_diff
                    elif in_channels_diff < 0:
                        module.groups = int(abs(in_channels_diff) / weight.shape[in_channels_axis])
                    module.in_channels += in_channels_diff
                if out_channels_diff != 0:
                    padding[-1] = out_channels_diff
            module.set_weights(name, ops.pad(weight, padding))
        return None


@ClassFactory.register('double_channels', 'BatchNorm2d')
class BatchNorm2dDoubleChannelArchitecture(Architecture):
    """Double Channel."""

    @staticmethod
    def decode(value, org_value):
        """Decode arch params."""
        return org_value

    @staticmethod
    def fit_weights(module, x):
        """Fit weights shape."""
        inputs = x[0] if isinstance(x, tuple) else x
        num_features_diff = 0
        for name, weight in module.get_weights().items():
            num_features_diff = int(inputs.shape[1]) - int(weight.shape[0])
            if num_features_diff == 0:
                continue
            padding = [0, num_features_diff]
            module.set_weights(name, ops.pad(weight, padding))
        if module.num_features:
            module.num_features += num_features_diff
        return None


@ClassFactory.register('double_channels', 'Linear')
class LinearDoubleChannelArchitecture(Architecture):
    """Double Channel."""

    @staticmethod
    def decode(value, org_value):
        """Decode arch params."""
        return org_value

    @staticmethod
    def fit_weights(module, x):
        """Fit weights shape."""
        inputs = x[0] if isinstance(x, tuple) else x
        in_features_diff = 0
        for name, weight in module.get_weights().items():
            if 'kernel' in name or 'weight' in name:
                in_features_diff = int(inputs.shape[1]) - int(weight.shape[1 if is_torch_backend() else 0])
                if in_features_diff == 0:
                    continue
                padding = [0, in_features_diff] if is_torch_backend() else [0, in_features_diff, 0, 0]
                module.set_weights(name, ops.pad(weight, padding))
        if module.in_features:
            module.in_features += in_features_diff
        return None


@ClassFactory.register('double_channels', 'Add')
class AddArchitecture(Architecture):
    """Double Channel."""

    @staticmethod
    def fit_weights(module, x):
        """Fit weights shape."""
        inputs = x[0] if isinstance(x, tuple) else x
        fit_weights_shapes = []
        for out_channels in module.out_channels:
            if not out_channels:
                fit_weights_shapes = [inputs.shape[1]]
            else:
                fit_weights_shapes.append(out_channels)
        fit_weights_shape = min(fit_weights_shapes)
        for child in module.children():
            if isinstance(child, ops.MaxPool2d):
                fit_weights_shape = inputs.shape[1]
        for child in module.children():
            if isinstance(child, ops.Conv2d):
                convs = [child]
            else:
                convs = [v for n, v in child.named_modules() if isinstance(v, ops.Conv2d)]
            if convs and convs[-1].out_channels != fit_weights_shape:
                convs[-1].out_channels = int(fit_weights_shape)
