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
