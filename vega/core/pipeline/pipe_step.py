# -*- coding:utf-8 -*-

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

    def __new__(cls, *args, **kwargs):
        """Create pipe step instance by ClassFactory."""
        t_cls = ClassFactory.get_cls(ClassType.PIPE_STEP, PipeStepConfig.type)
        return super().__new__(t_cls)

    def do(self, *args, **kwargs):
        """Do the main task in this pipe step."""
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
