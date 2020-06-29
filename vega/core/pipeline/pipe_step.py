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
from vega.core.common import TaskOps, UserConfig, FileOps
from vega.core.common.class_factory import ClassFactory, ClassType

logger = logging.getLogger(__name__)


class PipeStep(object):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self):
        self.task = TaskOps(UserConfig().data.general)

    def __new__(cls):
        """Create pipe step instance by ClassFactory."""
        t_cls = ClassFactory.get_cls(ClassType.PIPE_STEP)
        return super().__new__(t_cls)

    def do(self):
        """Do the main task in this pipe step."""
        raise NotImplementedError

    def _backup_output_path(self):
        # TODO: only backup step output path
        backup_path = self.task.backup_base_path
        if backup_path is None:
            return
        FileOps.copy_folder(self.task.local_output_path, backup_path)
