# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.


"""Extend HyperParameter classes."""
import math
import random
from itertools import permutations
from collections import Iterable, defaultdict
import numpy as np
from .param_types import ParamTypes
from .hyper_parameter import HyperParameter


class IntHyperParameter(HyperParameter):
    """Init HyperParameter."""

    param_type = ParamTypes.INT
    is_integer = True

    def cast(self, value):
        """Cast value.

        :param value: input `value`.
        :return: casted `value`.
        :rtype: float.

        """
        if value is not None:
            return int(value)
        else:
            return None

    def decode(self, x, forbidden=''):
        """Inverse transform.

        :param x: input `x`.
        :param str forbidden: forbidden, default is empty.
        :return: inverse transform to real `x`.

        """
        return x.astype(int)


class FloatHyperParameter(HyperParameter):
    """Float HyperParameter."""

    param_type = ParamTypes.FLOAT

    def cast(self, value):
        """Cast value.

        :param value: input `value`.
        :return: casted `value`.
        :rtype: float.

        """
        if value is not None:
            return float(value)
        else:
            return None


class FloatExpHyperParameter(HyperParameter):
    """Float Exp HyperParameter."""

    param_type = ParamTypes.FLOAT_EXP

    def cast(self, value):
        """Cast value.

        :param value: input `value`.
        :return: casted `value`.
        :rtype: float.

        """
        if value is not None:
            return math.log10(float(value))
        else:
            return None

    def encode(self, x, y=None):
        """Fit transform.

        :param x: input `x`.
        :param y: input `y`.
        :return: transform real `x` to hp's `x`.

        """
        x = x.astype(float)
        return np.log10(x)

    def decode(self, x, forbidden=''):
        """Inverse transform.

        :param x: input `x`.
        :param str forbidden: forbidden, default is empty.
        :return: inverse transform `x` back to real `x`.

        """
        x_power = np.power(10.0, x)
        x_power = min(max(x_power, self._param_range[0]), self._param_range[1])
        return x_power


class IntExpHyperParameter(HyperParameter):
    """Int Exp HyperParameter."""

    param_type = ParamTypes.INT_EXP

    def cast(self, value):
        """Cast value.

        :param value: input `value`.
        :return: casted `value`.
        :rtype: float.

        """
        if value is not None:
            return math.log10(float(value))
        else:
            return None

    def encode(self, x, y=None):
        """Fit transform.

        :param x: input `x`.
        :param y: input `y`.
        :return: transform real `x` to hp's `x`.

        """
        x = x.astype(float)
        return np.log10(x)

    def decode(self, x, forbidden=''):
        """Inverse transform.

        :param x: input `x`.
        :param str forbidden: forbidden, default is empty.
        :return: inverse transform `x` back to real `x`.

        """
        x_power = np.power(10.0, x).astype(int)
        x_power = min(max(x_power, self._param_range[0]), self._param_range[1])
        return x_power


class CatHyperParameter(HyperParameter):
    """Base class for Category HyperParameter.

    :param str param_name: hp's name, default name is `param`.
    :param int param_slice: slice count of hp, default is `0`.
    :param ParamTypes param_type: the type of hp, use `ParamTypes`.
    :param list param_range: the range list of hp.

    """

    param_type = ParamTypes.CATEGORY

    def __init__(self, param_name='param', param_slice=0, param_type=None, param_range=None, generator=None,
                 sample_num=None):
        """Init CategoryHyperParameter."""
        super(CatHyperParameter, self).__init__(param_name, param_slice, param_type, param_range, generator, sample_num)
        self.list_values = []
        self.cat_transform = {}
        # Converting array to index map
        for idx, each in enumerate(self.range):
            if isinstance(each, list):
                key = idx
                self.list_values.append(each)
            else:
                key = each
            if len(self.range) > 1:
                self.cat_transform[key] = idx / (len(self.range) - 1)
            else:
                self.cat_transform[key] = 0
        self.range = [0.0, 1.0]

    def multi_sample(self, param_range):
        """Sample multi values."""
        return [list(c) if isinstance(c, tuple) else c for c in permutations(param_range, self.sample_num)]

    def cast(self, value):
        """Cast value.

        :param value: input `value`.
        :raise: NotImplementedError

        """
        if isinstance(value, np.int64):
            return int(value)
        return value

    def check_legal(self, value):
        """Check value's legal.

        :param value: input `value`.
        :return: if value type is valid.
        :rtype: bool.

        """
        # print("cat check_legal")
        if self.cast(value) in self.cat_transform:
            return True
        else:
            return False

    def encode(self, x, y=None):
        """Fit transform.

        :param x: input `x`.
        :return: transform real `x` to hp's `x`.

        """
        # Accumulate the scores of each category
        # and the number of times that we have used it
        tmp_cat_transform = {each: (0, 0) for each in self.cat_transform.keys()}
        for i in range(len(x)):
            tmp_cat_transform[x[i]] = (
                tmp_cat_transform[x[i]][0] + y[i],  # sum score
                tmp_cat_transform[x[i]][1] + 1  # count occurrences
            )

        # If we have at least one score, compute the average
        for key, value in tmp_cat_transform.items():
            if value[1] != 0:
                self.cat_transform[key] = value[0] / float(value[1])
            else:
                self.cat_transform[key] = 0

        # Compute the range using the min and max scores
        range_max = max(
            self.cat_transform.keys(),
            key=(lambda k: self.cat_transform[k])
        )

        range_min = min(
            self.cat_transform.keys(),
            key=(lambda k: self.cat_transform[k])
        )

        self.range = [
            self.cat_transform[range_min],
            self.cat_transform[range_max]
        ]

        return np.vectorize(self.cat_transform.get)(x)

    def decode(self, x, forbidden=[]):
        """Inverse transform.

        :param x: input `x`.
        :param str forbidden: forbidden, default is empty.
        :return: inverse transform `x` back to real `x`.

        """
        # Compute the inverse dictionary
        inv_map = defaultdict(list)
        for key, value in self.cat_transform.items():
            if key not in forbidden:
                inv_map[value].append(key)
        keys = np.fromiter(inv_map.keys(), dtype=float)

        def invert(x):
            diff = (np.abs(keys - x))
            min_diff = diff[0]
            max_key = keys[0]

            # Find the score which is closer to the given value
            for i in range(len(diff)):
                if diff[i] < min_diff:
                    min_diff = diff[i]
                    max_key = keys[i]
                elif diff[i] == min_diff and keys[i] > max_key:
                    min_diff = diff[i]
                    max_key = keys[i]

            # Get a random category from the ones that had the given score
            return random.choice(np.vectorize(inv_map.get)(max_key))

        if isinstance(x, Iterable):
            transformed = list(map(invert, x))
            if isinstance(x, np.ndarray):
                transformed = np.array(transformed)

        else:
            transformed = self.cast(invert(x))
        if self.list_values:
            transformed = self.list_values[transformed]
        return transformed


class BoolCatHyperParameter(CatHyperParameter):
    """Bool Category HyperParameter."""

    param_type = ParamTypes.BOOL

    def cast(self, value):
        """Cast value.

        :param value: input `value`.
        :return: casted `value`.
        :rtype: float.

        """
        if value is not None:
            return bool(value)
        else:
            return None


class AdjacencyListHyperParameter(CatHyperParameter):
    """Bool Category HyperParameter."""

    param_type = ParamTypes.ADJACENCY_LIST

    def __init__(self, param_name='param', param_slice=0, param_type=None, param_range=None, generator=None,
                 sample_num=None):
        super(AdjacencyListHyperParameter, self).__init__(param_name, param_slice, param_type, param_range,
                                                          'AdjacencyList', sample_num)


class BinaryCodeHyperParameter(HyperParameter):
    """Int BinaryCode HyperParameter."""

    param_type = ParamTypes.BINARY_CODE

    def __init__(self, param_name='param', param_slice=0, param_type=None, param_range=None, generator=None,
                 sample_num=None):
        super(BinaryCodeHyperParameter, self).__init__(param_name, param_slice, param_type, param_range)

    def cast(self, value):
        """Cast value.

        :param value: input `value`.
        :return: casted `value`.
        :rtype: float.

        """
        return value

    def decode(self, x, forbidden=''):
        """Inverse transform.

        :param x: input `x`.
        :param str forbidden: forbidden, default is empty.
        :return: inverse transform to real `x`.
        """
        individual = []
        if len(self.range) == 1:
            prob = random.uniform(0.8, 0.95)
            for _ in range(self.range[0]):
                s = random.uniform(0, 1)
                if s > prob:
                    individual.append(0)
                else:
                    individual.append(1)
        else:
            if len(self.range) == 2:
                size = self.range[0]
                times = self.range[1]
                change_ids = random.sample(range(size), times)
                individual = [1 if i in change_ids else 0 for i in range(size)]
        return individual


class HalfCodeHyperParameter(HyperParameter):
    """Init HalfCode HyperParameter."""

    param_type = ParamTypes.HALF

    def __init__(self, param_name='param', param_slice=0, param_type=None, param_range=None, generator=None,
                 sample_num=None):
        super(HalfCodeHyperParameter, self).__init__(param_name, param_slice, param_type, param_range)

    def cast(self, value):
        """Cast value.

        :param value: input `value`.
        :return: casted `value`.
        :rtype: float.

        """
        return value

    def decode(self, x, forbidden=''):
        """Inverse transform.

        :param x: input `x`.
        :param str forbidden: forbidden, default is empty.
        :return: inverse transform to real `x`.
        """
        individual = []
        size = self.range[0]
        # TODO: TEST ONLY
        from vega.core.pipeline.conf import PipeStepConfig
        ratio = 0.8
        if hasattr(PipeStepConfig.search_space, "prune_ratio"):
            ratio = 1 - float(PipeStepConfig.search_space.prune_ratio)
        if random.uniform(0, 1) < ratio:
            return [1] * size
        if len(self.range) == 1:
            need_convert_code_size = size // 2
            change_ids = random.sample(range(size), need_convert_code_size)
            individual = [1 if i in change_ids else 0 for i in range(size)]
        return individual
