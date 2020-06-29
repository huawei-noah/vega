# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torch operators."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vega.core.common.class_factory import ClassType, ClassFactory


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Input(nn.Module):
    """Create Input for forward x."""

    def __init__(self, key=None, data=None, cuda=False):
        self.key = key
        self.cuda = cuda
        self.data = data
        super(Input, self).__init__()

    def forward(self, x):
        """Forward x."""
        if self.data:
            return self.data
        if self.key is not None:
            x = x[self.key]
        if self.cuda:
            if torch.is_tensor(x):
                return x.cuda()
            else:
                if isinstance(x, dict):
                    x = {key: value.cuda() if torch.is_tensor(x) else value for key, value in x.items()}
                else:
                    x = [item.cuda() if torch.is_tensor(x) else item for item in x]
        return x


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Lambda(nn.Module):
    """Create Lambda for forward x."""

    def __init__(self, func, data=None):
        self.func = func
        self.data = data
        super(Lambda, self).__init__()

    def forward(self, x):
        """Forward x."""
        if self.data:
            return self.func(self.data)
        out = self.func(x)
        return out


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Reshape(nn.Module):
    """Create Lambda for forward x."""

    def __init__(self, *args):
        self.args = args
        super(Reshape, self).__init__()

    def forward(self, x):
        """Forward x."""
        return x.reshape(*self.args)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Rermute(nn.Module):
    """Create Lambda for forward x."""

    def __init__(self, *args):
        self.args = args
        super(Rermute, self).__init__()

    def forward(self, x):
        """Forward x."""
        return x.rermute(*self.args)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class View(nn.Module):
    """Call torch.view."""

    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        """Forward x."""
        return x.view(x.size(0), -1)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Zero(nn.Module):
    """Zero block."""

    def __init__(self, stride):
        """Construct Zero class.

        :param stride: stride of the output
        """
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        """Do an inference on Zero.

        :param x: input tensor
        :return: output tensor
        """
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class FactorizedReduce(nn.Module):
    """Factorized reduce block."""

    def __init__(self, C_in, C_out, affine=True):
        """Construct FactorizedReduce class.

        :param C_in: input channel
        :param C_out: output channel
        :param affine: whether to use affine in BN
        """
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        """Do an inference on FactorizedReduce.

        :param x: input tensor
        :return: output tensor
        """
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Identity(nn.Module):
    """Class of Identity operation."""

    def __init__(self):
        """Init Identity."""
        super(Identity, self).__init__()

    def forward(self, x):
        """Forward function of Identity."""
        return x


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Pad(nn.Module):
    """Create Input for forward x."""

    def __init__(self, planes):
        super(Pad, self).__init__()
        self.planes = planes

    def forward(self, x):
        """Forward x."""
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes // 4, self.planes // 4), "constant", 0)


def drop_path(x, drop_prob):
    """Drop path operation.

    :param x: input feature map
    :type x: torch tensor
    :param drop_prob: dropout probability
    :type drop_prob: float
    :return: output feature map after dropout
    :rtype: torch tensor
    """
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x
