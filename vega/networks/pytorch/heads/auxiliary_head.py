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

"""AuxiliaryHead for NAGO."""
import torch.nn as nn
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.NETWORK)
class AuxiliaryHead(nn.Module):
    """Auxiliary head."""

    def __init__(self, C, num_classes, large_images):
        """Assuming input size 8x8 if large_images then the input will be 14x14."""
        super(AuxiliaryHead, self).__init__()
        k = 4
        if large_images:
            k = 7
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(k, stride=k, padding=0),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        """Forward method."""
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
