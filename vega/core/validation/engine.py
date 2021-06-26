# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Provide Validation engine."""
from functools import wraps
from vega.common import ClassFactory, ClassType


def validation(class_name):
    """Register the class to be verified.

    获取class_name对象，拦截其在validation中定义的属性并进行校验
    :param class_name: class name
    :return: wrapper
    """

    def decorator(cls):
        """Provide input param to decorator.

        :param func: wrapper function
        :return: decoratpr
        """
        # TODO： 需要导入包
        if isinstance(class_name, str):
            need_validate_cls = ClassFactory.get_cls(ClassType.CONFIG, class_name)
        else:
            need_validate_cls = class_name

        @wraps(cls)
        def wrapper(*args, **kwargs):
            """Make function as a wrapper."""
            valid_attrs = {key: item for key, item in cls.__dict__.items() if not key.startswith('_')}
            for attr_name, rules in valid_attrs.items():
                attr_value = getattr(need_validate_cls, attr_name)
                if isinstance(rules, list) or isinstance(rules, tuple):
                    for _rule in rules:
                        _rule(attr_value)
                else:
                    rules(attr_value)

            return cls(*args, **kwargs)

        return wrapper

    return decorator
