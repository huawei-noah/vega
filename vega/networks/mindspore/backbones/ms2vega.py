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

"""Transform torchvision model to vega."""
from collections import OrderedDict
import mindspore.nn as nn
from mindspore.ops import operations as P
from vega.modules.operators import ops
from vega.modules.blocks.blocks import BottleneckBlock
from vega.modules.connections import Sequential
from vega.networks.network_desc import NetworkDesc
from .resnet import ResidualBlock

atom_op = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU, nn.MaxPool2d, nn.AvgPool2d, P.ReduceMean,
           nn.Dense, nn.Dropout, nn.Flatten)
atom_block = (ResidualBlock)


def _transform_op(init_layer):
    """Transform the torch op to Vega op."""
    if isinstance(init_layer, nn.Conv2d):
        in_channels = init_layer.in_channels
        out_channels = init_layer.out_channels
        kernel_size = init_layer.kernel_size[0]
        stride = init_layer.stride
        padding = init_layer.padding
        new_layer = ops.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
    elif isinstance(init_layer, nn.BatchNorm2d):
        num_features = init_layer.num_features
        new_layer = ops.BatchNorm2d(num_features=num_features)
    elif isinstance(init_layer, nn.ReLU):
        new_layer = ops.Relu()
    elif isinstance(init_layer, nn.MaxPool2d):
        kernel_size = init_layer.kernel_size
        stride = init_layer.stride
        new_layer = ops.MaxPool2d(kernel_size=kernel_size, stride=stride)
    elif isinstance(init_layer, nn.AvgPool2d):
        kernel_size = init_layer.kernel_size
        stride = init_layer.stride
        padding = init_layer.padding
        new_layer = ops.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif isinstance(init_layer, P.ReduceMean):
        new_layer = ops.AdaptiveAvgPool2d()
    elif isinstance(init_layer, nn.Dense):
        in_features = init_layer.in_channels
        out_features = init_layer.out_channels
        new_layer = ops.Linear(in_features=in_features, out_features=out_features)
    elif isinstance(init_layer, nn.Dropout):
        prob = init_layer.p
        inplace = init_layer.inplace
        new_layer = ops.Dropout(prob=prob, inplace=inplace)
    elif isinstance(init_layer, nn.Flatten):
        new_layer = ops.View()
    else:
        raise ValueError("The op {} is not supported.".format(type(init_layer)))
    return new_layer


def _transform_block(init_block):
    """Transform the resnet block to Vega block."""
    if isinstance(init_block, ResidualBlock):
        inplanes = init_block.conv1.in_channels
        planes = init_block.bn1.num_features
        downsample = init_block.down_sample
        stride = 2 if downsample else 1
        new_block = BottleneckBlock(inchannel=inplanes, outchannel=planes, stride=stride)
    else:
        raise ValueError("Only support Bottleneck and BasicBlock, but got {}.".format(type(init_block)))
    return new_block


def transform_model(model):
    """Transform the torch model to Vega model."""
    new_model_dict = OrderedDict()
    for name, module in model.name_cells().items():
        if isinstance(module, atom_op):
            if isinstance(module, nn.Flatten):
                new_model_dict["mean"] = ops.AdaptiveAvgPool2d()
            new_model_dict[name] = _transform_op(module)

        sub_modules = OrderedDict()
        if isinstance(module, nn.SequentialCell):
            for sub_name, sub_module in module.name_cells().items():
                if isinstance(sub_module, atom_block):
                    sub_modules[sub_name] = _transform_block(sub_module)
                    sub_modules[sub_name].update_parameters_name(name + "." + sub_name + ".")
                if isinstance(sub_module, atom_op):
                    sub_modules[sub_name] = _transform_op(sub_module)
                    sub_modules[sub_name].update_parameters_name(name + "." + sub_name + ".")
            new_model_dict[name] = Sequential(sub_modules)
    model = Sequential(new_model_dict)
    desc = model.to_desc()
    vega_model = NetworkDesc(desc).to_model()
    return vega_model
