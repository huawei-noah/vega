# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for preprocess."""
from vega.modules.module import Module
from vega.common import ClassFactory, ClassType
from vega.modules.operators import ops


@ClassFactory.register(ClassType.NETWORK)
class PreOneStem(Module):
    """Class of one stem convolution.

    :param desc: description of PreOneStem
    :type desc: Config
    """

    def __init__(self, init_channels, stem_multi):
        """Init PreOneStem."""
        super(PreOneStem, self).__init__()
        self._c_curr = init_channels * stem_multi
        self.conv2d = ops.Conv2d(3, self._c_curr, 3, padding=1, bias=False)
        self.batchNorm2d = ops.BatchNorm2d(self._c_curr)

    @property
    def output_channel(self):
        """Get Output channel."""
        return self._c_curr

    def call(self, x):
        """Forward function of PreOneStem."""
        x = super().call(x)
        return x, x
