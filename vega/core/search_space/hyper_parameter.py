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


"""HyperParameter."""
import numpy as np
from .param_types import ParamTypes
from .range_generator import RangeGenerator


class HyperParameter(object):
    """Base HyperParameter class.

    :param str param_name: hp's name.
    :param int param_slice: slice count of hp, default is `0`.
    :param ParamTypes param_type: the type of hp, use `ParamTypes`.
    :param list param_range: the range list of hp.

    """

    param_type = None
    is_integer = False

    def cast(self, value):
        """Cast value, Base method.

        :param value: `value`.
        :raise: NotImplementedError

        """
        raise NotImplementedError()

    def __init__(self, param_name='param', param_slice=0, param_type=None,
                 param_range=None, generator=None, sample_num=None):
        """Init HyperParameter."""
        # maintain original param_range
        self.name = param_name
        if ('CAT' in param_type.name) | ('BOOL' in param_type.name):
            self.slice = len(param_range)
        else:
            self.slice = param_slice
        self._param_range = param_range
        self.sample_num = sample_num
        self.generator = generator
        if generator is not None:
            param_range = RangeGenerator(generator).create(param_range)
        # sample N without repeating, arrange all possible combinations.
        if sample_num is not None:
            param_range = self.multi_sample(param_range)
        self.range = [self.cast(value) for value in param_range]

    def get_name(self):
        """Get current hp's name.

        :return: name of hp.
        :rtype: str

        """
        return self.name

    def multi_sample(self, param_range):
        """Sample multi values."""
        return param_range

    def encode(self, x, y=None):
        """Encode input, Base method，need subclass to implement.

        :param x: intput `x`.
        :param y: intput `y`.
        :return: x.
        :rtype: x

        """
        return x

    def decode(self, x, forbidden=''):
        """Inverse transform, Base method, need subclass to implement.

        :param x: intput `x`.
        :param forbidden: intput forbidden name.
        :return: x.
        :rtype: x

        """
        return x

    def check_legal(self, value):
        """Check value's legal.

        :param value: input `value`.
        :return: if value type is valid.
        :rtype: bool.

        """
        return self.cast(value) >= self.range[0] and self.cast(value) <= self.range[-1]

    def compare(self, value_a, value_b):
        """Compare 2 values.

        :param value_a: input `value_a`.
        :param value_b: input `value_b`.
        :return: 0(equal), -1(a<b), 1(a>b).
        :rtype: int.

        """
        if value_a == value_b:
            return 0
        if value_a < value_b:
            return -1
        if value_a > value_b:
            return 1

    def get_grid_axis(self, grid_size):
        """Get grid axis.

        :param int grid_size: grid size.
        """
        if self.is_integer:
            return np.round(
                np.linspace(self.range[0], self.range[1], grid_size)
            )

        return np.round(
            np.linspace(self.range[0], self.range[1], grid_size),
            decimals=5,
        )

    def __eq__(self, other):
        """If self is equal to other."""
        if isinstance(self, other.__class__):
            _result = (self.param_type is other.param_type) and (self.is_integer == other.is_integer) and (
                    self.range == other.range)
            return _result
        return NotImplemented

    def __getnewargs__(self):
        """Get new args."""
        return (self.param_type, self._param_range)

    def allow_greater_less_comparison(self):
        """If allow greater less comparison.

        :return: if allow greater less comparison.
        :rtype: bool

        """
        return self.param_type != ParamTypes.BOOL and self.param_type != ParamTypes.CATEGORY

    def sample(self, n=1, decode=True, handler=None):
        """Random sample one hyper-param."""
        if len(self.range) == 1:
            low, high = 0, self.range[0]
        else:
            low, high = self.range
        if handler:
            low, high = handler(low, high)
        if self.is_integer:
            value = np.random.randint(low, high + 1, size=n)
        else:
            d = high - low
            value = low + d * np.random.rand(n)
        if decode:
            value = self.decode(value)
        return value
