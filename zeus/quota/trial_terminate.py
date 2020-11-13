# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Run Duration Termination."""
from zeus.common import ClassFactory, ClassType
from zeus.common.general import General
from .filter_terminate_base import FilterTerminateBase


@ClassFactory.register(ClassType.QUOTA)
class TrialTerminate(FilterTerminateBase):
    """Determine whether to terminate duration."""

    def __init__(self):
        super(TrialTerminate, self).__init__()
        self.max_trial = self.restrict_config.trials.get(General.step_name, None)
        self.count_trial = 0

    def is_halted(self, *args, **kwargs):
        """Halt or not."""
        if self.max_trial is None:
            return False
        self.count_trial += 1
        if self.count_trial > self.max_trial:
            return True
        else:
            return False
