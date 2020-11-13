# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Run Duration Termination."""
import time
from zeus.common import ClassFactory, ClassType
from zeus.common.general import General
from .filter_terminate_base import FilterTerminateBase


@ClassFactory.register(ClassType.QUOTA)
class DurationTerminate(FilterTerminateBase):
    """Determine whether to terminate duration."""

    def __init__(self):
        super(DurationTerminate, self).__init__()
        self.max_duration = self.restrict_config.duration.get(General.step_name, None)
        self.step_start_time = time.time()

    def is_halted(self, *args, **kwargs):
        """Halt or not."""
        if self.max_duration is None:
            return False
        current_time = time.time()
        duration = (current_time - self.step_start_time) / 3600
        if duration > self.max_duration:
            return True
        else:
            return False
