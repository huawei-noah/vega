# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Default parameter classes."""
import random
import numpy as np
from .base import Param
from modnas.registry.params import register


def _default_categorical_sampler(dim):
    return np.random.randint(dim)


def _default_int_sampler(bound):
    return random.randint(*bound)


def _default_real_sampler(bound):
    return random.uniform(*bound)


@register
class Categorical(Param):
    """Categorical parameter class."""

    TYPE = 'C'

    def __init__(self, choices, sampler=None, name=None, space=None, on_update=None):
        super().__init__(name, space, on_update)
        self.sample = _default_categorical_sampler if sampler is None else sampler
        self.choices = choices
        self._length = None
        self.val = None

    def extra_repr(self):
        """Return extra representation string."""
        return 'choices={}'.format(self.choices)

    def is_valid(self, value):
        """Return if the value is valid."""
        return value in self.choices

    def get_value(self, index):
        """Return value for given index."""
        return self.choices[index]

    def set_value(self, value, index=None):
        """Set parameter value."""
        index = self.get_index(value) if index is None else index
        self.val = index

    def value(self):
        """Return parameter value."""
        return self.choices[self.index()]

    def index(self):
        """Return parameter index."""
        if self.val is None:
            self.val = self.sample(len(self.choices))
        return self.val

    def get_index(self, value):
        """Return parameter index for given value."""
        return self.choices.index(value)

    def set_index(self, index):
        """Set parameter index."""
        self.set_value(index=index)

    def __len__(self):
        """Return choice size."""
        if self._length is None:
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
