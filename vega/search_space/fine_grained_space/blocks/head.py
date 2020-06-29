# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for head."""
from vega.search_space.fine_grained_space.fine_grained_space import FineGrainedSpace
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.fine_grained_space.operators import op, View


@ClassFactory.register(ClassType.SEARCH_SPACE)
class LinearClassificationHead(FineGrainedSpace):
    """Create LinearClassificationHead SearchSpace."""

    def constructor(self, base_channel, num_class):
        """Create layers.

        :param base_channel: base_channel
        :type base_channel: int
        :param num_class: number of class
        :type num_class: int
        """
        self.avgpool = op.AdaptiveAvgPool2d(output_size=(1, 1))
        self.view = View()
        self.Linear = op.Linear(in_features=base_channel, out_features=num_class)
