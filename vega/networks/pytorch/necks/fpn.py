# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""FPN neck for detection."""
from vega.common.class_factory import ClassType, ClassFactory
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


@ClassFactory.register(ClassType.NETWORK)
class FPN(FeaturePyramidNetwork):
    """Adds a FPN from torchvision."""

    def __init__(self, in_channels, out_channels=256):
        super(FPN, self).__init__(in_channels_list=in_channels, out_channels=out_channels,
                                  extra_blocks=LastLevelMaxPool())
