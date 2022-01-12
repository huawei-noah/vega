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

"""Swish activation functions."""
import torch.nn as nn
import torch.nn.functional as F
from modnas.registry.arch_space import register


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation function."""

    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """Return module output."""
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class HardSwish(nn.Module):
    """Hard Swish activation function."""

    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """Return module output."""
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Swish(nn.Module):
    """Swish activation function."""

    def forward(self, x):
        """Return module output."""
        return x * F.sigmoid(x)


register(HardSigmoid)
register(HardSwish)
register(Swish)
