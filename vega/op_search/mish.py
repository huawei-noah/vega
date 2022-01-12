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

"""This is a class for Mish."""
import torch.nn as nn
import torch
import torch.nn.functional as F


class Mish_init(nn.Module):
    """Define Mish init."""

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        """Forward mish."""
        return x * torch.tanh(F.softplus(x))


class Mish(nn.Module):
    """Define Mish."""

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        """Forward mish."""
        return x * torch.tanh(torch.log(torch.exp(x) + 1))


class Mish1(nn.Module):
    """Defiine new Mish."""

    def __init__(self):
        super(Mish1, self).__init__()

    def forward(self, x):
        """Forward mish."""
        return x * torch.tanh(torch.exp(x))
