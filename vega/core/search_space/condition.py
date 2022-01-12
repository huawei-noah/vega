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
