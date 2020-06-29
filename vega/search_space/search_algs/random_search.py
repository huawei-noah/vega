# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Random Search."""

import numpy as np
from vega.core.hyperparameter_space import HyperParameter, ParamTypes
from vega.core.hyperparameter_space import DiscreteSpaceBuilder
from vega.core.common.class_factory import ClassFactory, ClassType
from .search_algorithm import SearchAlgorithm


class RandomSearchAlgorithm(object):
    """RandomSearchAlgorithm.

    :param search_space: the target search space.
    :type search_space: SearchSpace
    """

    def __init__(self, search_space):
        """Init for RandomSearchAlgorithm."""
        self.space_builder = DiscreteSpaceBuilder(search_space.search_space)
        self.discrete_space = self.space_builder.get_discrete_space()

    def search(self, n=1):
        """Search function.

        :param n: number of sample to propose, defaults to 1
        :type n: int, optional
        :return: list of sample(dict) that sample from search space.
        :rtype: list
        """
        sample_list = []
        for _ in range(n):
            parameters = self.discrete_space.get_sample_space()
            if parameters is None:
                return None
            predictions = np.random.rand(parameters.shape[0], 1)
            index = np.argmax(predictions)
            param = self.discrete_space.inverse_transform(parameters[index, :])
            sample = self.space_builder.sample_to_dict(param)
            sample_list.append(sample)
        if n == 1:
            return sample_list[0]
        else:
            return sample_list

    def update(self, *args):
        """Update function for RandomSearchAlgorithm, currently not need to implement."""
        return
