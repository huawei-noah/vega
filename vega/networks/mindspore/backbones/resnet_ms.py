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

"""Resnst model."""
from vega.common import ClassType, ClassFactory
from .resnet import ResNet, ResidualBlock

_block_size = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}
in_channels = {
    50: [64, 256, 512, 1024],
    101: [64, 256, 512, 1024],
    152: [64, 256, 512, 1024]
}
out_channels = {
    50: [256, 512, 1024, 2048],
    101: [256, 512, 1024, 2048],
    152: [256, 512, 1024, 2048]
}
strides = {
    50: [1, 2, 2, 2],
    101: [1, 2, 2, 2],
    152: [1, 2, 2, 2]
}


@ClassFactory.register(ClassType.NETWORK)
class ResNetMs(ResNet):
    """Resnet Model form mindspore modelzoo."""

    def __init__(self, resnet_size, num_classes):
        #
        super(ResNetMs, self).__init__(ResidualBlock,
                                       _block_size[resnet_size],
                                       in_channels[resnet_size],
                                       out_channels[resnet_size],
                                       strides[resnet_size],
                                       num_classes)
