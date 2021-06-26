# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""FocalLoss for unbalanced data."""

from vega.modules.module import Module
from vega.common import ClassType, ClassFactory
import torch
from torch import nn


@ClassFactory.register(ClassType.LOSS)
class DisLoss(Module):
    """DisLoss."""

    def __init__(self):
        super(DisLoss, self).__init__()

    def call(self, inputs, targets):
        """Compute loss.

        :param inputs: predict data.
        :param targets: true data.
        :return:
        """
        real_validity, fake_validity = inputs
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
            torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        return d_loss


@ClassFactory.register(ClassType.LOSS)
class GenLoss(Module):
    """GenLoss."""

    def __init__(self):
        super(GenLoss, self).__init__()

    def call(self, inputs, targets):
        """Compute loss.

        :param inputs: predict data.
        :param targets: true data.
        :return:
        """
        fake_validity = inputs
        g_loss = -torch.mean(fake_validity)
        return g_loss


@ClassFactory.register(ClassType.LOSS)
class GANLoss(Module):
    """F1 Loss for unbalanced data."""

    def __init__(self):
        super(GANLoss, self).__init__()
        self.dis_loss = DisLoss()
        self.gen_loss = GenLoss()
