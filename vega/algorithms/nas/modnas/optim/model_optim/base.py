# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Score model optimum finder."""
import random


class ModelOptim():
    """Score model optimum finder class."""

    def __init__(self, space):
        self.space = space

    def get_random_index(self, excludes):
        """Return random categorical index from search space."""
        index = random.randint(0, self.space.categorical_size() - 1)
        while index in excludes:
            index = random.randint(0, self.space.categorical_size() - 1)
        return index

    def get_random_params(self, excludes):
        """Return random categorical parameters from search space."""
        return self.space.get_categorical_params(self.get_random_index(excludes))

    def get_optimums(self, model, size, excludes):
        """Return optimums in score model."""
        raise NotImplementedError
