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
import vega


def _get_data_format():
    if vega.is_torch_backend() or vega.is_ms_backend():
        return 'channels_first'
    elif vega.is_tf_backend():
        return 'channels_last'
    else:
        return None


def get_shape(layer):
    """Get weight shape."""
    if vega.is_tf_backend():
        return layer.get_shape()
    elif vega.is_torch_backend():
        return layer.weight.data.shape
    elif vega.is_ms_backend():
        para_name = list(layer._params)[0]
        return getattr(layer, para_name).default_input.shape


def _is_ops_instance(layer, name):
    """Get weight shape."""
    if vega.is_tf_backend():
        return layer.name.find(name) > 0
    else:
        return layer.__class__.__name__ == name


def _get_named_modules(layer):
    """Get named modules."""
    if vega.is_tf_backend():
        return [(op.name, op) for op in layer]
    elif vega.is_torch_backend():
        return layer.named_modules()
    elif vega.is_ms_backend():
        return layer._children_scope_recursive()


def _parse_module_name(name, module):
    """Parse the module name of mindspore."""
    if vega.is_ms_backend():
        while (list(module.cells()) != []):
            module = list(module.cells())[0]

        name_list = name.split("/")[1:]
        new_name = ""
        for name in name_list:
            name = "." + name.split("-")[0]
            new_name += name
        return new_name[1:], module
    else:
        return name, module


class PruneConv2D(object):
    """Prune Conv2D."""

    def __init__(self, layer):
        self.layer = layer
        shape = get_shape(layer)
        self.mask = np.ones(shape)

    def _make_mask(self, end_channel_idx, start_channel_idx=None):
        if _get_data_format() != 'channels_first':
            if start_channel_idx is not None:
                self.mask[:, :, start_channel_idx, :] = 0
            self.mask[:, :, :, end_channel_idx] = 0
        else:
            if start_channel_idx is not None:
                self.mask[:, start_channel_idx, :, :] = 0
            self.mask[end_channel_idx, :, :, :] = 0
        return self.mask

    def apply(self, end_mask_code, start_mask_code=None):
        """Apply mask to weight."""
        end_mask_code = np.array(end_mask_code)
        if start_mask_code is not None:
            start_mask_code = np.array(start_mask_code)
        start_channel_idx = None
        end_channel_idx = np.squeeze(np.argwhere(np.asarray(np.ones(end_mask_code.shape) - end_mask_code))).tolist()
        if start_mask_code is not None:
            start_channel_idx = np.squeeze(
                np.argwhere(np.asarray(np.ones(start_mask_code.shape) - start_mask_code))).tolist()
        self._make_mask(end_mask_code, start_mask_code)
        if vega.is_tf_backend():
            import tensorflow as tf
            return tf.assign(self.layer, self.layer * tf.constant(self.mask, dtype=self.layer.dtype))
        elif vega.is_torch_backend():
            import torch
            self.layer.weight.data = self.layer.weight.data * torch.FloatTensor(self.mask)
            self.layer.weight.data[end_channel_idx, :, :, :].requires_grad = False
            if start_channel_idx is not None:
                self.layer.weight.data[:, start_channel_idx, :, :].requires_grad = False
        elif vega.is_ms_backend():
            from mindspore import Tensor
            self.layer.weight.default_input = self.layer.weight.default_input * \
                Tensor(self.mask, self.layer.weight.default_input.dtype)
            for idx in end_channel_idx:
                self.layer.weight.default_input[idx, :, :, :].requires_grad = False
            if start_channel_idx is not None:
                for idx in start_channel_idx:
                    self.layer.weight.default_input[:, idx, :, :].requires_grad = False


class PruneBatchNorm(object):
    """Prune BatchNorm."""

    def __init__(self, layer):
        self.layer = layer
        self.mask = np.ones(get_shape(layer))

    def _make_mask(self, idx):
        self.mask[idx] = 0
        return self.mask

    def apply(self, mask_code):
        """Apply mask to batchNorm."""
        end_mask = np.asarray(mask_code)
        idx = np.squeeze(np.argwhere(np.asarray(np.ones(end_mask.shape) - end_mask))).tolist()
        self._make_mask(idx)
        if vega.is_tf_backend():
            import tensorflow as tf
            return tf.assign(self.layer, self.layer * tf.constant(self.mask, dtype=self.layer.dtype))
        elif vega.is_torch_backend():
            import torch
            self.layer.weight.data = self.layer.weight.data * torch.FloatTensor(self.mask)
            self.layer.bias.data = self.layer.bias.data * torch.FloatTensor(self.mask)
            self.layer.running_mean = self.layer.running_mean * torch.FloatTensor(self.mask)
            self.layer.running_var = self.layer.running_var * torch.FloatTensor(self.mask)
            self.layer.weight.data[idx].requires_grad = False
            self.layer.bias.data[idx].requires_grad = False
            self.layer.running_mean[idx].requires_grad = False
            self.layer.running_var[idx].requires_grad = False
        elif vega.is_ms_backend():
            from mindspore import Tensor
            self.layer.moving_mean.default_input = self.layer.moving_mean.default_input * \
                Tensor(self.mask, self.layer.moving_mean.default_input.dtype)
            self.layer.moving_variance.default_input = self.layer.moving_variance.default_input * \
                Tensor(self.mask, self.layer.moving_variance.default_input.dtype)
            self.layer.gamma.default_input = self.layer.gamma.default_input * \
                Tensor(self.mask, self.layer.gamma.default_input.dtype)
            self.layer.beta.default_input = self.layer.beta.default_input * \
                Tensor(self.mask, self.layer.beta.default_input.dtype)
            for id in idx:
                self.layer.moving_mean.default_input[id].requires_grad = False
                self.layer.moving_variance.default_input[id].requires_grad = False
                self.layer.gamma.default_input[id].requires_grad = False
                self.layer.beta.default_input[id].requires_grad = False


class PruneLinear(object):
    """Prune Linear."""

    def __init__(self, layer):
        self.layer = layer
        self.mask = np.ones(get_shape(layer))

    def _make_mask(self, idx):
        if _get_data_format() != 'channels_first':
            self.mask[idx, :] = 0
        else:
            self.mask[:, idx] = 0
        return self.mask

    def apply(self, mask_code):
        """Apply mask to linear."""
        mask_code = np.asarray(mask_code)
        idx = np.squeeze(np.argwhere(np.asarray(np.ones(mask_code.shape) - mask_code))).tolist()
        self._make_mask(idx)
        if vega.is_tf_backend():
            import tensorflow as tf
            return tf.assign(self.layer, self.layer * tf.constant(self.mask, dtype=self.layer.dtype))
        elif vega.is_torch_backend():
            import torch
            self.layer.weight.data = self.layer.weight.data * torch.FloatTensor(self.mask)
            self.layer.weight.data[:, idx].requires_grad = False
        elif vega.is_ms_backend():
            self.layer.weight.default_input = self.layer.weight.default_input * \
                torch.FloatTensor(self.mask, self.layer.weight.default_input.dtype)
            for id in idx:
                self.layer.weight.default_input[:, id].requires_grad = False


class PruneResnet(object):
    """Prune Resnet."""

    def __init__(self, layer):
        self.layer = layer

    def apply(self, chn_node_mask, chn_mask):
        """Apply mask to resnet."""
        end_mask = []
        for name, m1 in _get_named_modules(self.layer):
            name, m1 = _parse_module_name(name, m1)
            if name.startswith('backbone.init_block'):
                if name.endswith('conv'):
                    end_mask = chn_node_mask[0]
                    PruneConv2D(m1).apply(end_mask)
                elif name.startswith('bn'):
                    PruneBatchNorm(m1).apply(end_mask)
            elif name.startswith('backbone.layers'):
                parsed_name = list(name.split('.'))
                if len(parsed_name) <= 3:
                    continue
                block_idx = int(parsed_name[2][-1])
                layer_idx = block_idx + 1
                if _is_ops_instance(m1, 'Conv2d'):
                    if int(parsed_name[4]) == 0 and parsed_name[5].startswith('conv1'):
                        start_mask = chn_node_mask[layer_idx - 1]
                        end_mask = chn_mask[block_idx]
                        block_idx += 1
                    elif int(parsed_name[4]) == 0 and parsed_name[5].startswith('conv2'):
                        start_mask = end_mask
                        end_mask = chn_node_mask[layer_idx]
                    elif int(parsed_name[4]) == 1 and parsed_name[5].startswith('conv1'):
                        start_mask = chn_node_mask[layer_idx - 1]
                        end_mask = chn_node_mask[layer_idx]
                    PruneConv2D(m1).apply(end_mask, start_mask)
                elif _is_ops_instance(m1, 'BatchNorm2d'):
                    PruneBatchNorm(m1).apply(end_mask)
                elif _is_ops_instance(m1, 'Linear'):
                    PruneLinear(m1).apply(end_mask)
        return self.layer


class PruneMobileNet(PruneResnet):
    """Prune MobileNet."""

    def __init__(self, layer):
        super(PruneMobileNet, self).__init__(layer)

    def apply(self, chn_mask):
        """Apply mask to resnet."""
        end_mask = []
        for idx, (name, m1) in enumerate(_get_named_modules(self.layer)):
            name, m1 = _parse_module_name(name, m1)
            if name.startswith('features'):
                if len(name.split('.')) == 3:
                    module_length = len(m1._modules)
                elif len(name.split('.')) == 4:
                    sequence_idx = (int(name.split('.')[-3]) - 1) * 2
                    block_idx = int(name.split('.')[-1])

                    if block_idx < 2:
                        start_mask = chn_mask[sequence_idx - 1] if sequence_idx > 0 else None
                        end_mask = chn_mask[sequence_idx]
                    elif block_idx < module_length - 2:
                        if _is_ops_instance(m1, 'Conv2d'):
                            continue
                        end_mask = chn_mask[sequence_idx]
                        start_mask = end_mask
                    else:
                        start_mask = end_mask
                        end_mask = chn_mask[sequence_idx + 1]

                    if _is_ops_instance(m1, 'Conv2d'):
                        PruneConv2D(m1).apply(end_mask, start_mask)
                    elif _is_ops_instance(m1, 'BatchNorm2d'):
                        PruneBatchNorm(m1).apply(end_mask)
            elif name.startswith('classifier') and _is_ops_instance(m1, 'Linear'):
                PruneLinear(m1).apply(end_mask)
        return self.layer
