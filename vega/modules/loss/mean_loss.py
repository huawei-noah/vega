# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""MeanLoss for data."""
from vega.modules.module import Module
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.LOSS)
class MeanLoss(Module):
    """MeanLoss Loss for data."""

    def __init__(self):
        super(MeanLoss, self).__init__()

    def call(self, inputs, targets):
        """Compute loss, mean() to average on multi-gpu."""
        return inputs.mean()
