# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""System const variable."""
from enum import Enum, unique


@unique
class ClusterMode(Enum):
    """Cluster Mode."""

    Single = 0
    LocalCluster = 1


@unique
class WorkerTypes(Enum):
    """WorkerTypes."""

    TRAINER = 1
    EVALUATOR = 2
    GPU_EVALUATOR = 3
    HAVA_D_EVALUATOR = 4
