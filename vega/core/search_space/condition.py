# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.


"""Condition class."""


class Condition(object):
    """Base Condition class.

    :param HyperParameter child: a child hp.
    :param HyperParameter parent: a parent hp.
    :param ConditionTypes condition_type: ConditionTypes.
    :param list condition_range: list of value in parent.
    """

    condition_type = None

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
