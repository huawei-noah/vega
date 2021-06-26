# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Resnst model."""
from .resnet import ResNet, ResidualBlock
from vega.common import ClassType, ClassFactory

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
