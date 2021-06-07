# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""System const variable."""
from enum import IntEnum, unique, Enum


@unique
class WorkerTypes(IntEnum):
    """WorkerTypes."""

    TRAINER = 1
    EVALUATOR = 2
    HOST_EVALUATOR = 3
    DeviceEvaluator = 5


@unique
class Status(Enum):
    """Status type."""

    unstarted = "unstarted"
    initializing = "initializing"
    running = "running"
    finished = "finished"
    unknown = "unknown"
    error = "error"
    stopped = "stopped"


DatatimeFormatString = "%Y-%m-%d %H:%M:%S"
