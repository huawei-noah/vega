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

"""ForbiddenEqualsClause class."""
from .hyper_parameter import HyperParameter


class ForbiddenEqualsClause(object):
    """Forbidden Equals Clause.

    :param str param_name: hp's name.
    :param value: hp's value.
    """

    def __new__(cls, param_name, value):
        """Build new class."""
        if not isinstance(param_name, HyperParameter):
            raise ValueError('Invalid param_name type {}, should be Hyperparameter type.'.format(type(param_name)))

        if not param_name.check_legal(value):
            raise ValueError('Illegal hyperparameter value {}'.format(value))

        return object.__new__(cls)

    def __init__(self, param_name, value):
        """Init ForbiddenEqualsClause, _dict: {'loss': 'hinge'}."""
        self.param_name = param_name
        self.value = value
        self._dict = {}
        self._dict[param_name.name] = value


class ForbiddenAndConjunction(object):
    """Forbidden And Conjunction.

    :param list forbidden_list: a list of forbiddens.
    """

    def __new__(cls, forbidden_list):
        """Build new class."""
        if not isinstance(forbidden_list, list):
            raise ValueError('Invalid forbidden_list type {}, should be List type.'.format(type(forbidden_list)))
        for forbidden in forbidden_list:
            if not isinstance(forbidden, ForbiddenEqualsClause):
                raise ValueError(
                    'Invalid forbidden of list type {}, should be '
                    'ForbiddenEqualsClause type.'.format(type(forbidden)))
        return object.__new__(cls)

    def __init__(self, forbidden_list):
        """Init ForbiddenAndConjunction, _forbidden_dict: {'penalty':'l1', 'loss': 'hinge'}."""
        self.forbiddens = forbidden_list
        self._forbidden_dict = {}
        for forbidden in forbidden_list:
            self._forbidden_dict.update(forbidden._dict)
