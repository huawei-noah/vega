# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Fake loss for mindspore."""
from mindspore.nn.cell import Cell


class FakeLoss(Cell):
    """Fake loss for mindspore."""

    def __init__(self):
        """Init FakeLoss."""
        super(FakeLoss, self).__init__()

    def construct(self, output, label):
        """Forward of fake loss."""
        return 0
