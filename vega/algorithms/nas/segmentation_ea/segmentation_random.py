# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Random search algorithm used to search BiSeNet code."""

import numpy as np
from .conf import SegmentationConfig


class SegmentationRandom(object):
    """Random algorithm of SegmentationNas."""

    config = SegmentationConfig()

    def __init__(self, search_space=None):
        """Construct the SegmentationRandom class.

        :param search_space: Config of the search space
        """
        self.context_length = self.config.context_path.num_blocks
        self.double_channel = self.config.context_path.num_double_channel
        self.context_num_stage = self.config.context_path.num_stage
        self.spatial_length = self.config.spatial_path.num_blocks
        self.spatial_num_stages = self.config.spatial_path.num_stages

    def random_context_generator(self, length=10, num_reduction=3, num_stage=3):
        """Generate a random code of BiSeNet's context path.

        :param length: Length of the BiSeNet's context path.
        :param num_reduction: number of the reduction block.
        :param num_stage: inserted number of stages.
        :return: r18_64_ + code of context path.
        """
        arc = [1] * length
        position = np.random.choice(length, size=num_reduction, replace=False)
        for p in position:
            arc[p] = 2
        insert = np.random.choice(length - 1, size=num_stage - 1, replace=False)
        insert = [i + 1 for i in insert]
        insert = reversed(sorted(insert))
        for i in insert:
            arc.insert(i, '-')
        arc_string = ''.join(str(a) for a in arc)
        return 'r18_64_' + arc_string

    def random_spatial_generator(self, length, num_stage):
        """Generate a random code of BiSeNet's spatial path.

        :param length: Length of the BiSeNet's spatial path.
        :param num_stage: inserted number of stages.
        :return: code of spatial path.
        """
        inner_channel = np.random.choice([32, 64])
        arc = [1] * length
        position = np.random.choice(length, size=64 // inner_channel, replace=False)
        for p in position:
            arc[p] = 2
        insert = np.random.choice(length - 1, size=num_stage - 1, replace=False)
        insert = [i + 1 for i in insert]
        insert = reversed(sorted(insert))
        for i in insert:
            arc.insert(i, '-')
        arc_string = ''.join(str(a) for a in arc)
        arc_string = str(inner_channel) + '_' + arc_string
        return arc_string

    def search(self):
        """Generate a random code of BiSeNet.

        :return: code of the generated BiSeNet.
        """
        context_path = self.random_context_generator(self.context_length, self.double_channel, self.context_num_stage)
        spatial_path = self.random_spatial_generator(self.spatial_length, self.spatial_num_stages)
        encoding = [context_path, spatial_path]
        return encoding
