# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Simple CNN network."""

import torch.nn as nn
from vega.core.common.utils import update_dict
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import UserConfig, TaskOps, FileOps
from vega.search_space.networks import NetTypes, NetworkFactory, NetworkDesc
from vega.search_space.search_algs import SearchAlgorithm


@NetworkFactory.register(NetTypes.CUSTOM)
class SimpleCnn(nn.Module):
    """Simple CNN network."""

    def __init__(self, desc):
        """Initialize."""
        super(SimpleCnn, self).__init__()
        self.conv_num = 3
        self.conv_layers = nn.ModuleList([None] * self.conv_num)
        self.bn_layers = nn.ModuleList([None] * self.conv_num)
        self.relu_layers = nn.ModuleList([None] * self.conv_num)
        self.pool_layers = nn.ModuleList([None] * self.conv_num)
        conv_layer_names = ["conv_layer_{}".format(i) for i in range(self.conv_num)]
        inp_filters = 3
        out_size = 32
        for i, key in enumerate(conv_layer_names):
            out_filters = desc[key]['filters']
            kernel_size = desc[key]['kernel_size']
            padding = (kernel_size - 1) // 2
            self.conv_layers[i] = nn.Conv2d(inp_filters, out_filters, padding=padding, kernel_size=kernel_size)
            if 'bn' in desc[key].keys():
                if desc[key]['bn']:
                    self.bn_layers[i] = nn.BatchNorm2d(out_filters)
            if 'relu' in desc[key].keys():
                if desc[key]['relu']:
                    self.relu_layers[i] = nn.ReLU(inplace=False)
            self.pool_layers[i] = nn.MaxPool2d(2, stride=2)
            inp_filters = out_filters
            out_size = out_size // 2
        fc_inp_size = inp_filters * out_size * out_size
        fc_out_size = desc['fully_connect']['output_unit']
        self.fc0 = nn.Linear(fc_inp_size, fc_out_size)
        self.fc0_relu = nn.ReLU(inplace=True)
        fc_inp_size = fc_out_size
        fc_out_size = 10
        self.fc1 = nn.Linear(fc_inp_size, fc_out_size)

    def forward(self, x):
        """Forward."""
        for i in range(self.conv_num):
            x = self.conv_layers[i](x)
            if self.bn_layers[i] is not None:
                x = self.bn_layers[i](x)
            if self.relu_layers[i] is not None:
                x = self.relu_layers[i](x)
            x = self.pool_layers[i](x)
        x = self.fc0(x.view(x.size(0), -1))
        x = self.fc0_relu(x)
        x = self.fc1(x)
        return x
