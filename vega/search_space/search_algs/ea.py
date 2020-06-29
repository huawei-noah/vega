# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""EvolutionAlgorithm."""

import numpy as np
from .search_algorithm import SearchAlgorithm
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class EvolutionAlgorithm(SearchAlgorithm):
    """EvolutionAlgorithm.

    :param search_space: User defined search space.
    :type search_space: SearchSpace
    """

    def __init__(self, search_space=None):
        """Init for EvolutionAlgorithm."""
        super(EvolutionAlgorithm, self).__init__(search_space)

    def crossover(self, ind0, ind1):
        """Cross over operation in EA algorithm.

        :param ind0: input list.
        :type ind0: list
        :param ind1: input list.
        :type ind1: list
        :return: the random cross over results.
        :rtype: list, list
        """
        self.length = min(len(ind0), len(ind1))
        two_idxs = np.random.randint(0, self.length, 2)
        start_idx, end_idx = np.min(two_idxs), np.max(two_idxs)
        a_copy = ind0.copy()
        b_copy = ind1.copy()
        a_copy[start_idx: end_idx] = b_copy[start_idx: end_idx]
        b_copy[start_idx: end_idx] = a_copy[start_idx: end_idx]
        return a_copy, b_copy

    def mutate(self, ind):
        """Mutate operation in EA algorithm.

        :param ind: input list.
        :type ind: list
        :return: the random mutate list.
        :rtype: list
        """
        two_idxs = np.random.randint(0, len(ind), 2)
        start_idx, end_idx = np.min(two_idxs), np.max(two_idxs)
        a_copy = ind.copy()
        for k in range(start_idx, end_idx):
            a_copy[k] = 1 - a_copy[k]
        return a_copy
