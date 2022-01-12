# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
