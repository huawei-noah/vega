# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torch operators."""
import torch.nn as nn
from inspect import isclass
from vega.core.common.class_factory import ClassType, ClassFactory
from vega.core.common.config import Config


def import_torch_operators():
    """Import search space operators from torch."""
    ops = Config()
    for _name in dir(nn):
        if _name.startswith("_"):
            continue
        _cls = getattr(nn, _name)
        if not isclass(_cls):
            continue
        ops[_name] = ClassFactory.register_cls(_cls, ClassType.SEARCH_SPACE)

    return ops


op = import_torch_operators()
