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

"""FPN neck for detection."""
from vega.common.class_factory import ClassType, ClassFactory
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


@ClassFactory.register(ClassType.NETWORK)
class FPN(FeaturePyramidNetwork):
    """Adds a FPN from torchvision."""

    def __init__(self, in_channels, out_channels=256):
        super(FPN, self).__init__(in_channels_list=in_channels, out_channels=out_channels,
                                  extra_blocks=LastLevelMaxPool())
