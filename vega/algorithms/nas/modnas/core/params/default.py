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

"""Default parameter classes."""

import random
from typing import Callable, List, Optional, Union, Any
import numpy as np
from modnas.registry.params import register
from modnas.core.param_space import ParamSpace
from .base import Param


def _default_categorical_sampler(dim: int) -> int:
    return np.random.randint(dim)


def _default_int_sampler(bound):
    return random.randint(*bound)


def _default_real_sampler(bound):
    return random.uniform(*bound)


@register
class Categorical(Param):
    """Categorical parameter class."""

    TYPE = 'C'

    def __init__(
        self, choices: List[Any], sampler: Optional[Callable[[int], int]] = None, name: Optional[str] = None,
        space: Optional[ParamSpace] = None, on_update: Optional[Callable[[int], None]] = None
    ) -> None:
        super().__init__(name, space, on_update)
        self.sample = _default_categorical_sampler if sampler is None else sampler
        self.choices = choices
        self._length = -1
        self.val = 0

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return 'choices={}'.format(self.choices)

    def is_valid(self, value):
        """Return if the value is valid."""
        return value in self.choices

    def get_value(self, index: int) -> Any:
        """Return value for given index."""
        return self.choices[index]

    def set_value(self, value: Any, index: Optional[int] = None) -> None:
        """Set parameter value."""
        index = self.get_index(value) if index is None else index
        self.val = index

    def value(self) -> Any:
        """Return parameter value."""
        return self.choices[self.index()]

    def index(self) -> int:
        """Return parameter index."""
        if self.val is None:
            self.val = self.sample(len(self.choices))
        return self.val

    def get_index(self, value: Any) -> int:
        """Return parameter index for given value."""
        return self.choices.index(value)

    def set_index(self, index):
        """Set parameter index."""
        self.set_value(index=index)

    def __len__(self) -> int:
        """Return choice size."""
        if self._length == -1:
            self._length = len(self.choices)
        return self._length


@register
class Numeric(Param):
    """Numeric parameter class."""

    TYPE = 'N'

    def __init__(self, low, high, ntype=None, sampler=None, name=None, space=None, on_update=None):
        super().__init__(name, space, on_update)
        self.bound = (low, high)
        self.ntype = 'i' if (all(isinstance(b, int) for b in self.bound) and ntype != 'r') else 'r'
        default_sampler = _default_int_sampler if self.ntype == 'i' else _default_real_sampler
        self.sample = default_sampler if sampler is None else sampler
        self.val = None

    def extra_repr(self):
        """Return extra representation string."""
        return 'type={}, range={}'.format(self.ntype, self.bound)

    def is_valid(self, value):
        """Return if the value is valid."""
        return self.bound[0] <= value <= self.bound[1]

    def set_value(self, value):
        """Set parameter value."""
        if not self.is_valid(value):
            raise ValueError('invalid numeric parameter value')
        self.val = value

    def value(self):
        """Return parameter value."""
        if self.val is None:
            self.val = self.sample(self.bound)
        return self.val

    def is_int(self):
        """Return if the parameter is integer-valued."""
        return self.ntype == 'i'

    def is_real(self):
        """Return if the parameter is real-valued."""
        return self.ntype == 'r'
