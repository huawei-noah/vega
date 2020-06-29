# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import darts related torch operators."""
import torch.nn as nn
import torch.nn.functional as F
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory


@NetworkFactory.register(NetTypes.Operator)
class Identity(nn.Module):
    """Class of Identity operation."""

    def __init__(self):
        """Init Identity."""
        super(Identity, self).__init__()

    def forward(self, x):
        """Forward function of Identity."""
        return x


@NetworkFactory.register(NetTypes.Operator)
class Zero(nn.Module):
    """Class of Zero operation.

    :param stride: stride of Zero
    :type stride: int
    """

    def __init__(self, stride):
        """Init Zero."""
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        """Forward Function fo Zero."""
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)
