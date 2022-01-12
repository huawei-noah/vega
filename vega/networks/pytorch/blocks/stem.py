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

"""This is SearchSpace for preprocess."""
import torch.nn as nn
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.NETWORK)
class PreTwoStem(nn.Module):
    """Class of two stems convolution."""

    def __init__(self, init_channels):
        """Init PreTwoStem."""
        super(PreTwoStem, self).__init__()
        self._C = init_channels
        self.stems = nn.ModuleList()
        stem0 = nn.Sequential(
            nn.Conv2d(3, self._C // 2, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(self._C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self._C // 2, self._C, 3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(self._C),
        )
        self.stems += [stem0]
        stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self._C, self._C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self._C),
        )
        self.stems += [stem1]
        self.C_curr = self._C

    @property
    def output_channel(self):
        """Get Output channel."""
        return self._C

    def forward(self, x):
        """Forward function of PreTwoStem."""
        out = [x]
        for stem in self.stems:
            out += [stem(out[-1])]
        return out[-2], out[-1]
