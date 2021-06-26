# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Load the offical model from torchvision."""

output_layer_map = {
    "resnet18": "layer4",
    "resnet34": "layer4",
    "resnet50": "layer4",
    "resnet101": "layer4",
    "resnet152": "layer4",
    "vgg16": "features",
    "vgg19": "features",
    "alexnet": "features",
    "googlenet": "inception5b",
    "mobilenet_v2": "features",
    "inception_v3": "Mixed_7c",
    "densenet121": "features",
    "densenet169": "features",
    "densenet201": "features",
}
