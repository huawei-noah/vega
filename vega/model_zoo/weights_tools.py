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

"""Model zoo."""
import re
from collections import OrderedDict
from vega.modules.operators import ops


def convert_names(model):
    """Convert Names."""
    names = []
    for k, v in model.named_modules():
        if isinstance(v, ops.Conv2d):
            names.append(k + '.weight')
        elif isinstance(v, ops.BatchNorm2d):
            names.append(k + '.running_mean')
            names.append(k + '.running_var')
            names.append(k + '.weight')
            names.append(k + '.bias')
        elif isinstance(v, ops.Linear):
            names.append(k + '.weight')
            names.append(k + '.bias')
    return names


def convert_faster_backbone_weights(model, state_dict):
    """Convert resnet weights from torchvision to Faster Backbone weights name."""
    names = convert_names(model)
    new_state_dict = OrderedDict()
    for name in names:
        state_name = name
        state_name = state_name.replace('backbone.res_layers_seq.0', 'layer1')
        state_name = state_name.replace('backbone.res_layers_seq.1', 'layer2')
        state_name = state_name.replace('backbone.res_layers_seq.2', 'layer3')
        state_name = state_name.replace('backbone.res_layers_seq.3', 'layer4')
        state_name = state_name.replace('backbone.conv1', 'conv1')
        state_name = state_name.replace('backbone.norm1', 'bn1')
        state_name = state_name.replace('norm', 'bn')
        state_name = state_name.replace('head', 'fc')
        new_state_dict[name] = state_dict.pop(state_name)
    if len(state_dict) == 0:
        return new_state_dict
    else:
        raise ValueError('Failed to convert weigh of faster_backbone.')


def convert_resnet_general_weights(model, state_dict):
    """Convert resnet weights from torchvision to vega weights name."""
    names = convert_names(model)
    new_state_dict = OrderedDict()
    replace_mapping = {'backbone.layers.BottleneckBlock0.': 'layer1.0.',
                       'backbone.layers.BottleneckBlock1.': 'layer1.1.',
                       'backbone.layers.BottleneckBlock2.': 'layer1.2.',
                       'backbone.layers.BottleneckBlock3.': 'layer2.0.',
                       'backbone.layers.BottleneckBlock4.': 'layer2.1.',
                       'backbone.layers.BottleneckBlock5.': 'layer2.2.',
                       'backbone.layers.BottleneckBlock6.': 'layer2.3.',
                       'backbone.layers.BottleneckBlock7.': 'layer3.0.',
                       'backbone.layers.BottleneckBlock8.': 'layer3.1.',
                       'backbone.layers.BottleneckBlock9.': 'layer3.2.',
                       'backbone.layers.BottleneckBlock10.': 'layer3.3.',
                       'backbone.layers.BottleneckBlock11.': 'layer3.4.',
                       'backbone.layers.BottleneckBlock12.': 'layer3.5.',
                       'backbone.layers.BottleneckBlock13.': 'layer4.0.',
                       'backbone.layers.BottleneckBlock14.': 'layer4.1.',
                       'backbone.layers.BottleneckBlock15.': 'layer4.2.',
                       'backbone.init_block.conv': 'conv1',
                       'backbone.init_block.bn': 'bn1',
                       'backbone.init_block.batch': 'bn1', }
    for name in names:
        state_name = name
        for k, v in replace_mapping.items():
            state_name = state_name.replace(k, v)
        if state_name.startswith('head'):
            state_name = state_name.replace('head', 'fc')
        elif state_name.startswith('layer'):
            state_name = re.sub(r'block.0.', '', state_name)
            state_name = re.sub(r'block.1.', 'downsample.', state_name)
            if 'downsample' in state_name:
                state_name = re.sub(r'conv1', '0', state_name)
                state_name = re.sub(r'batch', '1', state_name)
            state_name = re.sub(r'batch', 'bn', state_name)
        new_state_dict[name] = state_dict.pop(state_name)
    if len(state_dict) == 0:
        return new_state_dict
    else:
        raise ValueError('Failed to convert weigh of resnet_general.')


def convert_torch_resnet_weights_to_serialClassificationNet(model, state_dict, strict=True):
    """Convert resnet weights from torchvision to vega weights name."""
    names = convert_names(model)
    new_state_dict = OrderedDict()
    for name in names:
        state_name = name
        state_name = state_name.replace('backbone.', '')
        if name.startswith('head.linear'):
            state_name = state_name.replace('head.linear', 'fc')
        elif state_name.startswith('layers'):
            state_name = re.sub(r'block.0.', '', state_name)
            state_name = re.sub(r'block.1.', 'downsample.', state_name)
            if 'downsample' in state_name:
                state_name = re.sub(r'conv1', '0', state_name)
                state_name = re.sub(r'batch', '1', state_name)
            state_name = re.sub(r'batch', 'bn', state_name)
            layer_no = int(state_name.split('.')[1])
            state_name = state_name.replace('layers.{}'.format(layer_no), 'layer{}'.format(layer_no + 1))
        new_state_dict[name] = state_dict.pop(state_name)
    if strict:
        if len(state_dict) != 0:
            raise ValueError('Failed to convert resnet weights to serialClassificationNet')
    return new_state_dict


def convert_fasterrcnn_fpn_weights(model, state_dict):
    """Convert resnet weights from torchvision to vega weights name."""
    state_dict = {k: v for k, v in state_dict.items() if k.startswith('backbone.fpn')}
    names = convert_names(model)
    new_state_dict = OrderedDict()
    for name in names:
        state_name = name
        if name.startswith('fpn_convs'):
            state_name = state_name.replace('fpn_convs', 'backbone.fpn.layer_blocks')
        elif name.startswith('lateral_convs'):
            state_name = state_name.replace('lateral_convs', 'backbone.fpn.inner_blocks')
        new_state_dict[name] = state_dict.pop(state_name)
    return new_state_dict


def convert_fasterrcnn_backbone_weights(model, state_dict):
    """Convert resnet weights from torchvision to vega weights name."""
    state_dict = {k.replace('backbone.body.', ''): v for k, v in state_dict.items()}
    return convert_torch_resnet_weights_to_serialClassificationNet(model, state_dict, False)
