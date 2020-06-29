# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Backbone of mobilenet v2."""
from torchvision.models import MobileNetV2
import torch
import torch.nn as nn


class MobileNetV2Backbone(MobileNetV2):
    """Backbone of mobilenet v2."""

    def __init__(self, load_path=None):
        """Construct MobileNetV3Tiny class.

        :param load_path: path for saved model
        """
        super(MobileNetV2Backbone, self).__init__()
        self.features = nn.ModuleList(list(self.features)[:18])

        if load_path is not None:
            self.load_state_dict(torch.load(load_path), strict=False)

    def forward(self, x):
        """Do an inference on MobileNetV2.

        :param x: input tensor
        :return: output tensor
        """
        outs = []
        for i, feature in enumerate(self.features):
            x = feature(x)
            if i in [3, 6, 13, 17]:
                outs.append(x)

        return outs
