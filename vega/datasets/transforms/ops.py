# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is the function of scale."""
PARAMETER_MAX = 10


def float_parameter(level, maxval):
    """Scale 'val' between 0 and maxval.

    :param level: Level of the operation that will be between [0, 'PARAMETER_MAX'].
    :type level: int
    :param maxval: Maximum value that the operation can have. This will be scaled to level/PARAMETER_MAX.
    :type maxval: int
    :return: results from scaling 'maxval' according to 'level'.
    :rtype: float
    """
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    """Scale 'val' between 0 and maxval.

    :param level: Level of the operation that will be between [0, 'PARAMETER_MAX'].
    :type level: int
    :param maxval: Maximum value that the operation can have. This will be scaled to level/PARAMETER_MAX.
    :type maxval: int
    :return: results from scaling 'maxval' according to 'level'.
    :rtype: int
    """
    return int(level * maxval / PARAMETER_MAX)
