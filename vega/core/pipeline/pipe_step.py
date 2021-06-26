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
from datetime import datetime
from vega.common import TaskOps, Status
from vega.common import ClassFactory, ClassType
from vega.core.pipeline.conf import PipeStepConfig
from vega.report import ReportServer


__all__ = ["PipeStep"]
logger = logging.getLogger(__name__)


class PipeStep(object):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self, name=None, **kwargs):
        """Initialize pipestep."""
        self.task = TaskOps()
        self.name = name if name else "pipestep"
        self.start_time = datetime.now()
        self.status = Status.unstarted
        self.message = None
        self.end_time = None
        self.num_epochs = None
        self.num_models = None
        # TODO
        # ReportServer().restore()

    def __new__(cls, *args, **kwargs):
        """Create pipe step instance by ClassFactory."""
        t_cls = ClassFactory.get_cls(ClassType.PIPE_STEP, PipeStepConfig.type)
        return super().__new__(t_cls)

    def do(self, *args, **kwargs):
        """Do the main task in this pipe step."""
        # set self.num_models, self.epochs and self.status=running/finished
        pass

    def save_info(self):
        """Save step info to report serve."""
        info = {"step_name": self.name}
        for attr in dir(self):
            if attr in ["start_time", "end_time", "status", "message", "num_epochs", "num_models"]:
                info[attr] = getattr(self, attr)
        ReportServer().update_step_info(**info)

    def update_status(self, status, desc=None):
        """Update step status."""
        if status == Status.finished:
            self.end_time = datetime.now()
        self.status = status
        self.message = desc
        self.save_info()
