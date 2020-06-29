# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.


"""Condition types."""
from enum import Enum


class ConditionTypes(Enum):
    """Condition's types."""

    EQUAL = 1
    NOT_EQUAL = 2
    IN = 3
    # LESS = 4
    # GREATER = 5


CONDITION_TYPE_MAP = {
    'EQUAL': ConditionTypes.EQUAL,
    'NOT_EQUAL': ConditionTypes.NOT_EQUAL,
    'IN': ConditionTypes.IN
}
