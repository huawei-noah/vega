# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.


"""Param types."""
from enum import Enum


class ParamTypes(Enum):
    """Define param types."""

    INT = 1
    INT_EXP = 2
    FLOAT = 3
    FLOAT_EXP = 4
    CATEGORY = 5
    BOOL = 6
    BINARY_CODE = 7
    ADJACENCY_LIST = 8
    HALF = 9


PARAM_TYPE_MAP = {
    'INT': ParamTypes.INT,
    'INT_EXP': ParamTypes.INT_EXP,
    'FLOAT': ParamTypes.FLOAT,
    'FLOAT_EXP': ParamTypes.FLOAT_EXP,
    'CATEGORY': ParamTypes.CATEGORY,
    'BOOL': ParamTypes.BOOL,
    'BINARY_CODE': ParamTypes.BINARY_CODE,
    'ADJACENCY_LIST': ParamTypes.ADJACENCY_LIST,
    'HALF': ParamTypes.HALF
}
