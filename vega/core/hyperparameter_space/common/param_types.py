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
    INT_CAT = 3
    FLOAT = 4
    FLOAT_EXP = 5
    FLOAT_CAT = 6
    STRING = 7
    BOOL = 8


PARAM_TYPE_MAP = {
    'INT': ParamTypes.INT,
    'INT_EXP': ParamTypes.INT_EXP,
    'INT_CAT': ParamTypes.INT_CAT,
    'FLOAT': ParamTypes.FLOAT,
    'FLOAT_EXP': ParamTypes.FLOAT_EXP,
    'FLOAT_CAT': ParamTypes.FLOAT_CAT,
    'STRING': ParamTypes.STRING,
    'BOOL': ParamTypes.BOOL
}
