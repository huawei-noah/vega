# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Set env."""
import sys
import logging
from zeus.common import init_log
from zeus.common.general import General
from zeus.common.task_ops import TaskOps
from vega.core.run import init_local_cluster_args

logger = logging.getLogger(__name__)


def _init_env():
    if sys.version_info < (3, 6):
        sys.exit('Sorry, Python < 3.6 is not supported.')
    init_log(level=General.logger.level,
             log_path=TaskOps().local_log_path)
    General.env = init_local_cluster_args()
    _print_task_id()


def _print_task_id():
    logging.info("-" * 48)
    logging.info("  task id: {}".format(General.task.task_id))
    logging.info("-" * 48)
