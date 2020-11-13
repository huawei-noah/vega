# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Generator range values."""
from itertools import product


class RangeGenerator(object):
    """Base class for all range Generator."""

    def __new__(cls, name):
        """Create Range Generator class."""
        for sub_cls in cls.__subclasses__():
            if sub_cls.__name__ == name:
                return super().__new__(sub_cls)

    def create(self, range):
        """Generate a adjacency list."""
        raise NotImplementedError


class AdjacencyList(RangeGenerator):
    """Generator for create a adjacency list."""

    def create(self, range_value):
        """Create a adjacency list according to range.

        :param range: node list, like [1, 2, 3, 4]
        :return: all relations of each node, [[1, 2], [1, 3], [1, 4], [2,3], [2, 4], [3, 4]]
        """
        adjacency_list = []
        for idx, node in enumerate(range_value):
            if idx == len(range_value):
                break
            for node2 in range_value[idx + 1:]:
                adjacency_list.append([node, node2])
        return adjacency_list


class BinaryList(RangeGenerator):
    """Generator for create a binary list."""

    def create(self, range_value):
        """Create a binary list according to range.

        :param range: define list length, like [5]
        :return: all relations of each node, [0, 1, 1, 0, 1]
        """
        return product(range(2), repeat=range_value[0])
