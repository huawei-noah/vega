# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.


"""Extend different Condition classes."""
from .condition_types import ConditionTypes
from .condition import Condition
from .param_types import ParamTypes


class EqualCondition(Condition):
    """Equal Condition.

    :param HyperParameter child: a child hp.
    :param HyperParameter parent: a parent hp.
    :param ConditionTypes condition_type: ConditionTypes.
    :param list condition_range: list of value in parent.

    """

    condition_type = ConditionTypes.EQUAL

    def __init__(self, child, parent, condition_type, condition_range):
        """Init EqualCondition."""
        if len(condition_range) != 1:
            raise ValueError('Invalid condition_range {}, EqualCondition should'
                             ' only have one condition value.'.format(condition_range))
        super(EqualCondition, self).__init__(child, parent, condition_type, condition_range)

    def _evaluate(self, value):
        """Evaluate this condition.

        :param value: input `value`.
        :return: result of evaluate.
        :rtype: bool.
        """
        for cv in self.range:
            if self.parent.compare(value, cv) == 0:
                return True
        return False


class NotEqualCondition(Condition):
    """Not Equal Condition.

    :param HyperParameter child: a child hp.
    :param HyperParameter parent: a parent hp.
    :param ConditionTypes condition_type: ConditionTypes.
    :param list condition_range: list of value in parent.

    """

    condition_type = ConditionTypes.NOT_EQUAL

    def __init__(self, child, parent, condition_type, condition_range):
        """Init NotEqualCondition."""
        super(NotEqualCondition, self).__init__(child, parent, condition_type, condition_range)

    def _evaluate(self, value):
        """Evaluate this condition.

        :param value: input `value`.
        :return: result of evaluate.
        :rtype: bool.
        """
        for cv in self.range:
            if self.parent.compare(value, cv) == 0:
                return False
        return True


class InCondition(Condition):
    """In Condition.

    :param HyperParameter child: a child hp.
    :param HyperParameter parent: a parent hp.
    :param ConditionTypes condition_type: ConditionTypes.
    :param list condition_range: list of value in parent.

    """

    condition_type = ConditionTypes.IN

    def __init__(self, child, parent, condition_type, condition_range):
        """Init InCondition."""
        if parent.param_type in [ParamTypes.INT_CAT, ParamTypes.FLOAT_CAT,
                                 ParamTypes.BOOL, ParamTypes.STRING]:
            if len(condition_range) < 1:
                raise ValueError('Invalid condition_range {}, InCondition for {}'
                                 ' should at least have one condition value.'
                                 .format(condition_range, parent.param_type))
        elif len(condition_range) != 2:
            raise ValueError('Invalid condition_range {}, InCondition for {}'
                             ' should at least have two condition value.'
                             .format(condition_range, parent.param_type))
        elif condition_range[0] > condition_range[-1]:
            raise ValueError('Invalid condition_range {}, InCondition for {}'
                             ' should has a valid param range like [a, b] where a <= b.'
                             .format(condition_range, parent.param_type))
        super(InCondition, self).__init__(child, parent, condition_type, condition_range)

    def _evaluate(self, value):
        """Evaluate this condition.

        :param value: input `value`.
        :return: result of evaluate.
        :rtype: bool.
        """
        if self.parent.param_type in [ParamTypes.INT_CAT, ParamTypes.FLOAT_CAT,
                                      ParamTypes.BOOL, ParamTypes.STRING]:
            for cv in self.range:
                if self.parent.compare(value, cv) == 0:
                    return True
            return False
        else:
            low = self.range[0]
            high = self.range[-1]
            if self.parent.compare(low, value) != 1 and self.parent.compare(high, value) != -1:
                return True
            else:
                return False
