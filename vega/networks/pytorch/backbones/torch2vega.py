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
import torch.nn as nn
import torchvision
from vega.modules.operators import ops
from vega.networks.necks import Bottleneck, BasicBlock
from vega.modules.connections import Sequential
from vega.common import ClassType, ClassFactory
from vega.modules.module import Module

atom_op = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
           nn.Linear, nn.Dropout)
atom_block = (torchvision.models.resnet.Bottleneck, torchvision.models.resnet.BasicBlock)


def _transsorm_op(init_layer):
    """Transform the torch op to Vega op."""
    if isinstance(init_layer, nn.Conv2d):
        in_channels = init_layer.in_channels
        out_channels = init_layer.out_channels
        kernel_size = init_layer.kernel_size
        stride = init_layer.stride
        padding = init_layer.padding
        new_layer = ops.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
    elif isinstance(init_layer, nn.BatchNorm2d):
        num_features = init_layer.num_features
        new_layer = ops.BatchNorm2d(num_features=num_features)
    elif isinstance(init_layer, nn.ReLU):
        new_layer = ops.Relu()
    elif isinstance(init_layer, nn.MaxPool2d):
        kernel_size = init_layer.kernel_size
        stride = init_layer.stride
        padding = init_layer.padding
        new_layer = ops.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif isinstance(init_layer, nn.AvgPool2d):
        kernel_size = init_layer.kernel_size
        stride = init_layer.stride
        padding = init_layer.padding
        new_layer = ops.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif isinstance(init_layer, nn.AdaptiveAvgPool2d):
        output_size = init_layer.output_size
        new_layer = ops.AdaptiveAvgPool2d(output_size=output_size)
    elif isinstance(init_layer, nn.Linear):
        in_features = init_layer.in_features
        out_features = init_layer.out_features
        new_layer = ops.Linear(in_features=in_features, out_features=out_features)
    elif isinstance(init_layer, nn.Dropout):
        prob = init_layer.p
        inplace = init_layer.inplace
        new_layer = ops.Dropout(prob=prob, inplace=inplace)
    else:
        raise ValueError("The op {} is not supported.".format(type(init_layer)))
    return new_layer


def _transform_block(init_block):
    """Transform the resnet block to Vega block."""
    if isinstance(init_block, torchvision.models.resnet.Bottleneck):
        inplanes = init_block.conv1.in_channels
        planes = init_block.bn1.num_features
        stride = init_block.stride
        downsample = init_block.downsample
        dilation = init_block.conv2.dilation
        new_block = Bottleneck(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample,
                               dilation=dilation)
    elif isinstance(init_block, torchvision.models.resnet.BasicBlock):
        inplanes = init_block.conv1.in_channels
        planes = init_block.bn1.num_features
        stride = init_block.stride
        downsample = init_block.downsample
        dilation = init_block.conv2.dilation
        new_block = BasicBlock(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample,
                               dilation=dilation)
    else:
        raise ValueError("Only support Bottleneck and BasicBlock, but got {}.".format(type(init_block)))
    return new_block


def _transfowm_model(model):
    """Transform the torch model to Vega model."""
    new_model_dict = OrderedDict()
    for name, module in model.named_children():
        if isinstance(module, atom_op):
            new_model_dict[name] = _transsorm_op(module)

        sub_modules = OrderedDict()
        if isinstance(module, nn.Sequential):
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, atom_block):
                    sub_modules[sub_name] = _transform_block(sub_module)
                if isinstance(sub_module, atom_op):
                    sub_modules[sub_name] = _transsorm_op(sub_module)
            new_model_dict[name] = Sequential(sub_modules)
        return new_model_dict


class Torch2Vega(Module):
    """Transform torchvision model to vega."""

    def __init__(self, model_name=None, **kwargs):
        super(Torch2Vega).__init__(self)
        model = ClassFactory.get_cls(ClassType.NETWORK, 'torchvision_' + model_name)
        self.model = model(**kwargs) if kwargs else model()
        self.vega_model = _transfowm_model(self.model)

    def call(self, inputs):
        """Forward of the network."""
        return self.vega_model(inputs)
