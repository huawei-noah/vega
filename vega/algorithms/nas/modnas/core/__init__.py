# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from functools import wraps, partial


def make_decorator(func):
    """Return wrapped function that acts as decorator if no extra positional args are given."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        if len(args) == 0 and len(kwargs) > 0:
            return partial(func, *args, **kwargs)
        return func(*args, **kwargs)

    return wrapped


def singleton(cls):
    """Return wrapped class that has only one instance."""
    inst = []

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if not inst:
            inst.append(cls(*args, **kwargs))
        return inst[0]
    return get_instance
