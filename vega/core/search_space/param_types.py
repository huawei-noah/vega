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
