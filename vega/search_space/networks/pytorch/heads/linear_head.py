# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined LinearClassificationHead."""
import torch
import torch.nn as nn
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory


@NetworkFactory.register(NetTypes.HEAD)
class LinearClassificationHead(Network):
    """LinearClassificationHead."""

    def __init__(self, desc):
        """Init LinearClassificationHead."""
        super(LinearClassificationHead, self).__init__()
        self.net_desc = desc
        base_channel = desc["base_channel"]
        num_classes = desc["num_classes"]
        self.base_channel = base_channel
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(base_channel, num_classes)

    def forward(self, x):
        """Forward function of ClassificationHead."""
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.linear(x)
        return out

    @property
    def input_shape(self):
        """Get the model input tensor shape."""
        return

    @property
    def output_shape(self):
        """Get the model output tensor shape."""
        return (1, self.num_classes)

    @property
    def model_layers(self):
        """Get the model layers."""
        return 1
