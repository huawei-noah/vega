# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Provide wrapper functions."""
from inspect import signature as sig
from functools import wraps, partial
from vega.core.common.class_factory import ClassFactory


def metric(name=None):
    """Make function as a metrics, use the same params from configuration.

    :param func: source function
    :return: wrapper
    """

    def decorator(func):
        """Provide input param to decorator.

        :param func: wrapper function
        :return: decoratpr
        """
        setattr(func, 'name', name or func.__name__)

        @ClassFactory.register('trainer.metrics')
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Make function as a wrapper."""
            params_sig = sig(func).parameters
            params = {param: value for param, value in kwargs.items() if param in params_sig}
            return func(*args, **params)

        return wrapper

    return decorator
