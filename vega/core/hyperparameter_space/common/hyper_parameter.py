# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.


"""HyperParameter."""
import numpy as np
import six
from .param_types import ParamTypes


class HyperParameter(object):
    """Base HyperParameter class.

    :param str param_name: hp's name.
    :param int param_slice: slice count of hp, default is `0`.
    :param ParamTypes param_type: the type of hp, use `ParamTypes`.
    :param list param_range: the range list of hp.

    """

    param_type = None
    is_integer = False

    _subclasses = []

    @classmethod
    def _get_subclasses(cls):
        """Get subclasses.

        :return: list of subclasses.
        :rtype: list.

        """
        subclasses = []
        for subclass in cls.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(subclass._get_subclasses())

        return subclasses

    @classmethod
    def subclasses(cls):
        """Get subclasses.

        :return: list of subclasses.
        :rtype: list.

        """
        if not cls._subclasses:
            cls._subclasses = cls._get_subclasses()

        return cls._subclasses

    def __new__(cls, param_name='param', param_slice=0, param_type=None,
                param_range=None):
        """Build new class."""
        if not isinstance(param_name, str):
            raise ValueError(
                'Invalid param name {}, shold be str type.'.format(param_type))

        if (not isinstance(param_slice, int)) & (param_slice is not None):
            raise ValueError(
                'Invalid param slice {}, shold be int type.'.format(
                    param_slice))

        if not isinstance(param_type, ParamTypes):
            if isinstance(param_type, six.string_types) and \
                    param_type.upper() in ParamTypes.__members__:
                param_type = ParamTypes[param_type.upper()]
            else:
                raise ValueError('Invalid param type {}'.format(param_type))

        for subclass in cls.subclasses():
            if subclass.param_type is param_type:
                return super(HyperParameter, cls).__new__(subclass)

    def cast(self, value):
        """Cast value, Base method.

        :param value: `value`.
        :raise: NotImplementedError

        """
        raise NotImplementedError()

    def __init__(self, param_name='param', param_slice=0, param_type=None,
                 param_range=None):
        """Init HyperParameter."""
        # maintain original param_range
        self.name = param_name

        if ('CAT' in param_type.name) | (
                'STRING' in param_type.name) | (
                'BOOL' in param_type.name):
            self.slice = len(param_range)
        else:
            self.slice = param_slice

        self._param_range = param_range

        self.range = [self.cast(value) for value in param_range]

    def get_name(self):
        """Get current hp's name.

        :return: name of hp.
        :rtype: str

        """
        return self.name

    def fit_transform(self, x, y):
        """Fit transform, Base method，need subclass to implement.

        :param x: intput `x`.
        :param y: intput `y`.
        :return: x.
        :rtype: x

        """
        return x

    def inverse_transform(self, x, forbidden=''):
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
        if self.cast(value) >= self.range[0] and \
                self.cast(value) <= self.range[-1]:
            return True
        else:
            return False

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
            _result = self.param_type is other.param_type and \
                self.is_integer == other.is_integer and self.range == other.range
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
        if self.param_type != ParamTypes.BOOL \
                and self.param_type != ParamTypes.STRING:
            return True
        else:
            return False
