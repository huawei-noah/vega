# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for blocks."""
from vega.search_space.fine_grained_space import FineGrainedSpace
from vega.search_space.fine_grained_space.conditions import Sequential
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.fine_grained_space.operators import conv_bn_relu6
from vega.search_space.fine_grained_space.blocks.sr import InvertedResidual


@ClassFactory.register(ClassType.SEARCH_SPACE)
class MobileNetV3Tiny(FineGrainedSpace):
    """Create MobileNetV3Tiny SearchSpace."""

    inverted_residual_setting = [
        [1.0, 9, 1],
        [4.0, 14, 2],
        [3.0, 14, 1],
        [3.0, 24, 2],
        [3.0, 24, 1],
        [3.0, 24, 1],
        [6.0, 48, 2],
        [2.5, 48, 1],
        [2.3, 48, 1],
        [2.3, 48, 1],
        [6.0, 67, 1],
        [6.0, 67, 1],
        [6.0, 96, 2],
        [6.0, 96, 1],
        [6.0, 96, 1],
        [6.0, 96, 1]]

    def constructor(self, load_path=None):
        """Construct MobileNetV3Tiny class.

        :param load_path: path for saved model
        """
        input_channel = 9
        features = [conv_bn_relu6(inchannel=3, outchannel=input_channel, kernel=3, stride=2)]
        for _, lst in enumerate(self.inverted_residual_setting):
            output_channel = lst[1]
            features.append(InvertedResidual(inp=input_channel, oup=output_channel, stride=lst[2], expand_ratio=lst[0]))
            input_channel = output_channel
        self.block = Sequential(*tuple(features), out_list=[3, 6, 12, 16])
