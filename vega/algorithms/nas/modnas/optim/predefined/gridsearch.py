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

"""Basic categorical Optimizers."""
import time
import random
from collections import OrderedDict
from typing import Optional
from modnas.registry.optim import register
from modnas.core.param_space import ParamSpace
from ..base import CategoricalSpaceOptim


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

    def __init__(self, seed: Optional[int] = None, space: Optional[ParamSpace] = None) -> None:
        super().__init__(space)
        seed = int(time.time()) if seed is None else seed
        random.seed(seed)

    def _next(self) -> OrderedDict:
        index = self.get_random_index()
        self.visited.add(index)
        return self.space.get_categorical_params(index)
