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

"""Prune operators."""
import numpy as np
from vega import is_torch_backend


class PruneConv2DFilter(object):
    """Prune Conv2D."""

    def __init__(self, layer, props):
        self.layer = layer
        self.props = props
        self.start_mask_code = self.props.get(self.layer.name + '/in_channels')
        if self.start_mask_code:
            if len(self.start_mask_code) != self.layer.in_channels:
                raise ValueError('Start_mask_code must be equal to inchannels.')
        self.end_mask_code = self.props.get(self.layer.name + '/out_channels')
        if len(self.end_mask_code) != self.layer.out_channels:
            raise ValueError('End_mask_code must be equal to outchannels.')

    def filter(self):
        """Apply mask to weight."""
        if self.start_mask_code:
            self.filter_in_channels(self.start_mask_code)
        if self.end_mask_code:
            self.filter_out_channels(self.end_mask_code)

    def _make_mask(self, mask_code):
        """Make Mask by mask code."""
        if sum(mask_code) == 0:
            mask_code[0] = 1
        mask_code = np.array(mask_code)
        idx = np.squeeze(np.argwhere(mask_code)).tolist()
        idx = [idx] if not isinstance(idx, list) else idx
        return idx

    def filter_out_channels(self, mask_code):
        """Mask out channels."""
        filter_idx = self._make_mask(mask_code)
        weights = self.layer.get_weights()
        self.layer.out_channels = sum(mask_code)
        for name, weight in weights.items():
            if weight is not None:
                if is_torch_backend():
                    self.layer.set_weights(name, weight[filter_idx, :, :, :])
                else:
                    self.layer.set_weights(name, weight[:, :, :, filter_idx])

    def filter_in_channels(self, mask_code):
        """Mask in channels."""
        filter_idx = self._make_mask(mask_code)
        weights = self.layer.get_weights()
        self.layer.in_channels = sum(mask_code)
        for name, weight in weights.items():
            if weight is not None:
                if is_torch_backend():
                    self.layer.set_weights(name, weight[:, filter_idx, :, :])
                else:
                    self.layer.set_weights(name, weight[:, :, filter_idx, :])


class PruneBatchNormFilter(object):
    """Prune BatchNorm."""

    def __init__(self, layer, props):
        self.layer = layer
        self.props = props
        self.mask_code = self.props.get(self.layer.name + '/num_features')

    def filter(self):
        """Apply mask to batchNorm."""
        if sum(self.mask_code) == 0:
            self.mask_code[0] = 1
        mask_code = np.asarray(self.mask_code)
        idx = np.squeeze(np.argwhere(mask_code)).tolist()
        idx = [idx] if not isinstance(idx, list) else idx
        weights = self.layer.get_weights()
        self.layer.num_features = sum(mask_code)
        for name, weight in weights.items():
            self.layer.set_weights(name, weight[idx])


class PruneLinearFilter(object):
    """Prune Linear."""

    def __init__(self, layer, props):
        self.layer = layer
        self.props = props
        self.mask_code = self.props.get(self.layer.name + '/in_features')
        if len(self.mask_code) != self.layer.in_features:
            raise ValueError('Failed to init Prune Linear.')

    def filter(self):
        """Apply mask to linear."""
        if sum(self.mask_code) == 0:
            self.mask_code[0] = 1
        mask_code = np.asarray(self.mask_code)
        idx_in = np.squeeze(np.argwhere(mask_code)).tolist()
        idx_in = [idx_in] if not isinstance(idx_in, list) else idx_in
        self.layer.in_features = sum(mask_code)
        weights = self.layer.get_weights()
        out_size = self.layer.out_features
        for name, weight in weights.items():
            if 'kernel' in name or 'weight' in name:
                if is_torch_backend():
                    self.layer.set_weights(name, weight[:, idx_in])
                    out_size = weight.shape[0]
                else:
                    self.layer.set_weights(name, weight[idx_in, :])
                    out_size = weight.shape[1]
        # fineTune out_feature value
        if self.layer.out_features == out_size:
            return
        idx_out = list(np.random.permutation(out_size)[:self.layer.out_features])
        for name, weight in self.layer.get_weights().items():
            if 'kernel' in name:
                self.layer.set_weights(name, weight[:, idx_out])
            else:
                self.layer.set_weights(name, weight[idx_out])
        self.layer.out_features = out_size
