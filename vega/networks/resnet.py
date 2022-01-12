# -*- coding:utf-8 -*-

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

"""This is SearchSpace for network."""
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module
from vega.modules.operators.ops import Linear, AdaptiveAvgPool2d, View
from .resnet_general import ResNetGeneral


@ClassFactory.register(ClassType.NETWORK)
class ResNet(Module):
    """Create ResNet SearchSpace."""

    def __init__(self, depth=18, base_channel=64, out_plane=None, stages=4, num_class=10, small_input=True,
                 doublechannel=None, downsample=None):
        """Create layers.

        :param num_reps: number of layers
        :type num_reqs: int
        :param items: channel and stride of every layer
        :type items: dict
        :param num_class: number of class
        :type num_class: int
        """
        super(ResNet, self).__init__()
        self.backbone = ResNetGeneral(small_input, base_channel, depth, stages, doublechannel, downsample)
        self.adaptiveAvgPool2d = AdaptiveAvgPool2d(output_size=(1, 1))
        self.view = View()
        out_plane = out_plane or self.backbone.out_channels
        self.head = Linear(in_features=out_plane, out_features=num_class)
