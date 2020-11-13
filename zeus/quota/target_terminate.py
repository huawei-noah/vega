# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric Target Termination."""
from zeus.common import ClassFactory, ClassType
from .filter_terminate_base import FilterTerminateBase


@ClassFactory.register(ClassType.QUOTA)
class TargetTerminate(FilterTerminateBase):
    """Determine whether to satisfy target."""

    def __init__(self):
        super(TargetTerminate, self).__init__()
        self.target_type = self.target_config.type
        self.target_value = self.target_config.value

    def is_halted(self, *args, **kwargs):
        """Halt or not."""
        if self.target_type is None or self.target_value is None:
            return False
        valid_metric = kwargs[self.target_type]
        if valid_metric > self.target_value:
            return True
        else:
            return False
