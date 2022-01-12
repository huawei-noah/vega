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

"""Load the official model from mindspore modelzoo."""

output_layer_map = {
    "resnet50": "layer4",
    "resnet101": "layer4",
    "vgg16": "features",
    "alexnet": "features",
    "googlenet": "inception5b",
    "mobilenet_v1": "features",
    "mobilenet_v2": "features",
    "mobilenet_v3": "features",
    "inception_v3": "Mixed_7c",
    "inception_v4": "Mixed_7c",
    "densenet121": "features",
}

location_map = {
    "resnet50": ("cv.resnet.src.resnet", "resnet50"),
    "resnet101": ("cv.resnet.src.resnet", "resnet101"),
    "vgg16": ("cv.vgg16.src.vgg", "vgg16"),
    "alexnet": ("cv.alexnet.src.alexnet", "AlexNet"),
    "googlenet": ("cv.googlenet.src.googlenet", "GoogleNet"),
    "mobilenet_v1": ("cv.mobilenetv1.src.mobilenet_v1", "mobilenet_v1"),
    "mobilenet_v2": ("cv.mobilenetv2.src.mobilenet_v2", "mobilenet_v2"),
    "mobilenet_v3": ("cv.mobilenetv3.src.mobilenet_v3", "mobilenet_v3"),
    "inception_v3": ("cv.inceptionv3.src.inception_v3", "InceptionV3"),
    "inception_v4": ("cv.inceptionv4.src.inception_v4", "InceptionV4"),
    "densenet121": ("cv.densenet121.src.network.densenet", "DenseNet121"),
}
