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
import six
from .param_types import ParamTypes
from .hyper_parameter import HyperParameter
from .condition import Condition
from .condition_types import ConditionTypes


class ParamsFactory(object):
    """Base SearchSpace class.

    :param str param_name: hp's name.
    :param int param_slice: slice count of hp, default is `0`.
    :param ParamTypes param_type: the type of hp, use `ParamTypes`.
    :param list param_range: the range list of hp.

    """

    _subclasses_hp = []
    _subclasses_condition = []

    @classmethod
    def _get_subclasses(cls, base_class):
        """Get subclasses.

        :return: list of subclasses.
        :rtype: list.

        """
        subclasses = []
        for subclass in base_class.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(cls._get_subclasses(subclass))
        return subclasses

    @classmethod
    def params(cls):
        """Get subclasses.

        :return: list of subclasses.
        :rtype: list.

        """
        if not cls._subclasses_hp:
            cls._subclasses_hp = cls._get_subclasses(HyperParameter)
        return cls._subclasses_hp

    @classmethod
    def conditions(cls):
        """Get subclasses.

        :return: list of subclasses.
        :rtype: list.

        """
        if not cls._subclasses_condition:
            cls._subclasses_condition = cls._get_subclasses(Condition)
        return cls._subclasses_condition

    @classmethod
    def create_search_space(cls, param_name='param', param_slice=0, param_type=None, param_range=None, generator=None,
                            sample_num=None):
        """Build new class."""
        if not isinstance(param_name, str):
            raise ValueError(
                'Invalid param name {}, should be str type.'.format(param_type))

        if (not isinstance(param_slice, int)) & (param_slice is not None):
            raise ValueError(
                'Invalid param slice {}, should be int type.'.format(
                    param_slice))

        if not isinstance(param_type, ParamTypes):
            if isinstance(param_type, six.string_types) and \
                    param_type.upper() in ParamTypes.__members__:
                param_type = ParamTypes[param_type.upper()]
            else:
                raise ValueError('Invalid param type {}'.format(param_type))

        for subclass in cls.params():
            if subclass.param_type is param_type:
                return subclass(param_name, param_slice, param_type, param_range, generator, sample_num)

        raise ValueError("Not found search space class, type={}".format(param_type))

    @classmethod
    def create_condition(cls, child, parent, condition_type, condition_range):
        """Build new class."""
        if not isinstance(child, HyperParameter):
            raise ValueError('Invalid child type {}, '
                             'should be SearchSpace type.'.format(child))
        if not isinstance(parent, HyperParameter):
            raise ValueError('Invalid child type {}, '
                             'should be SearchSpace type.'.format(parent))

        if not isinstance(condition_type, ConditionTypes):
            if isinstance(condition_type, six.string_types) and \
                    condition_type.upper() in ConditionTypes.__members__:
                condition_type = ConditionTypes[condition_type.upper()]
            else:
                raise ValueError('Invalid param type {}'.format(condition_type))
        for cv in condition_range:
            if not parent.check_legal(cv):
                raise ValueError('Illegal condition value {}'.format(cv))
        for subclass in cls.conditions():
            if subclass.condition_type is condition_type:
                return subclass(child, parent, condition_type, condition_range)

        raise ValueError("Not found condition class, type={}".format(condition_type))

    @classmethod
    def is_params(cls, params):
        """Check whether a variable is a SearchSpace."""
        if not hasattr(params, "param_type"):
            return False
        for subclass in cls.params():
            if subclass.param_type is params.param_type:
                return True
        return False

    @classmethod
    def is_condition(cls, condition):
        """Check whether a variable is a condition."""
        if not hasattr(condition, "condition_type"):
            return False
        for subclass in cls.conditions():
            if subclass.condition_type is condition.condition_type:
                return True
        return False
