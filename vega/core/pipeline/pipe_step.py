# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""PipeStep that used in Pipeline."""
import logging
from zeus.common import TaskOps
from zeus.common import ClassFactory, ClassType
from vega.core.pipeline.conf import PipeStepConfig


logger = logging.getLogger(__name__)


class PipeStep(object):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self):
        self.task = TaskOps()

    def __new__(cls):
        """Create pipe step instance by ClassFactory."""
        t_cls = ClassFactory.get_cls(ClassType.PIPE_STEP, PipeStepConfig.type)
        return super().__new__(t_cls)

    def do(self):
        """Do the main task in this pipe step."""
        raise NotImplementedError
