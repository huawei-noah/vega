# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.


"""Condition class."""
import six
from .condition_types import ConditionTypes
from .hyper_parameter import HyperParameter


class Condition(object):
    """Base Condition class.

    :param HyperParameter child: a child hp.
    :param HyperParameter parent: a parent hp.
    :param ConditionTypes condition_type: ConditionTypes.
    :param list condition_range: list of value in parent.
    """

    condition_type = None

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

    def __new__(cls, child, parent, condition_type, condition_range):
        """Build new class."""
        if not isinstance(child, HyperParameter):
            raise ValueError('Invalid child type {}, '
                             'should be Hyperparameter type.'.format(child))
        if not isinstance(parent, HyperParameter):
            raise ValueError('Invalid child type {}, '
                             'should be Hyperparameter type.'.format(parent))

        if not isinstance(condition_type, ConditionTypes):
            if isinstance(condition_type, six.string_types) and \
                    condition_type.upper() in ConditionTypes.__members__:
                condition_type = ConditionTypes[condition_type.upper()]
            else:
                raise ValueError('Invalid param type {}'.format(condition_type))
        for cv in condition_range:
            if not parent.check_legal(cv):
                raise ValueError('Illegal condition value {}'.format(cv))
        for subclass in cls.subclasses():
            if subclass.condition_type is condition_type:
                return super(Condition, cls).__new__(subclass)

    def cast(self, value):
        """Cast value, Base method.

        :param value: `value`.
        :raise: NotImplementedError

        """
        raise NotImplementedError()

    def __init__(self, child, parent, condition_type, condition_range):
        """Init Condition class."""
        # maintain original param_range
        self.child = child
        self.parent = parent
        self._condition_range = condition_range
        self.range = [parent.cast(value) for value in condition_range]

    def evaluate(self, value):
        """Check legal of condition and evaluate this condition.

        :param value: input `value`.
        :return: result of evaluate.
        :rtype: bool.
        """
        if not self.parent.check_legal(value):
            raise ValueError('Ilegal evaluate value {}'.format(value))
        return self._evaluate(value)
