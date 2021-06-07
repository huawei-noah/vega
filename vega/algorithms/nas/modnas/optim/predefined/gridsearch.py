# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Basic categorical Optimizers."""
import time
import random
from ..base import CategoricalSpaceOptim
from modnas.registry.optim import register


@register
class GridSearchOptim(CategoricalSpaceOptim):
    """Optimizer using grid search."""

    def __init__(self, space=None):
        super().__init__(space)
        self.counter = 0

    def _next(self):
        index = self.counter
        self.counter = self.counter + 1
        return self.space.get_categorical_params(index)

    def has_next(self):
        """Return True if Optimizer has the next set of parameters."""
        return self.counter < self.space_size()


@register
class RandomSearchOptim(CategoricalSpaceOptim):
    """Optimizer using random search."""

    def __init__(self, seed=None, space=None):
        super().__init__(space)
        seed = int(time.time()) if seed is None else seed
        random.seed(seed)

    def _next(self):
        index = self.get_random_index()
        self.visited.add(index)
        return self.space.get_categorical_params(index)
