# -*- coding:utf-8 -*-

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

"""Score model optimum finder."""
import random
from collections import OrderedDict
from typing import Set


class ModelOptim():
    """Score model optimum finder class."""

    def __init__(self, space):
        self.space = space

    def get_random_index(self, excludes: Set[int]) -> int:
        """Return random categorical index from search space."""
        index = random.randint(0, self.space.categorical_size() - 1)
        while index in excludes:
            index = random.randint(0, self.space.categorical_size() - 1)
        return index

    def get_random_params(self, excludes: Set[int]) -> OrderedDict:
        """Return random categorical parameters from search space."""
        return self.space.get_categorical_params(self.get_random_index(excludes))

    def get_optimums(self, model, size, excludes):
        """Return optimums in score model."""
        raise NotImplementedError
