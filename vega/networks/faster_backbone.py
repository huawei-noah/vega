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
from .spnet_backbone import SpResNetDet


@ClassFactory.register(ClassType.NETWORK)
class FasterBackbone(Module):
    """Create ResNet SearchSpace."""

    def __init__(self, code=None, depth=18, base_channel=64, out_plane=2048, stage=4, num_class=1000, small_input=True,
                 block='BasicBlock', pretrained_arch=None, pretrained=None):
        """Create layers.

        :param num_reps: number of layers
        :type num_reqs: int
        :param items: channel and stride of every layer
        :type items: dict
        :param num_class: number of class
        :type num_class: int
        """
        super(FasterBackbone, self).__init__()
        self.backbone = SpResNetDet(depth=depth, block=block, code=code,
                                    pretrained=pretrained, pretrained_arch=pretrained_arch)
        self.adaptiveAvgPool2d = AdaptiveAvgPool2d(output_size=(1, 1))
        self.view = View()
        out_plane = out_plane or self.backbone.out_channels
        self.head = Linear(in_features=out_plane, out_features=num_class)

    def call(self, x, **kwargs):
        """Forward compute of resnet for detection."""
        out = self.backbone(x)
        out = self.adaptiveAvgPool2d(out[-1])
        out = self.view(out)
        out = self.head(out)
        return out
