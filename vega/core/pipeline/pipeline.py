# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Pipeline that string up all PipeSteps."""
import sys
import os
import traceback
import logging
import signal
from .pipe_step import PipeStep
from vega.core.common.user_config import UserConfig
from vega.core.common.class_factory import ClassFactory
from vega.core.scheduler.master import Master
from copy import deepcopy

logger = logging.getLogger(__name__)


class Pipeline(object):
    """Load configs and provide `run` method to start pipe steps.

    In this class, Pipeline will parse all pipe steps from the config data.
    Execute steps one by one and set glob configs with current step config.
    """

    def __init__(self):
        """Init Pipeline and set task_id and configs."""
        self.cfg = UserConfig().data
        # TODO: validate cfg by validator
        if self.cfg is None:
            raise ValueError("please load user config before pipeline init.")
        self._steps = self.cfg.pipeline
        logger.info("pipeline steps:%s", str(self._steps))

    def run(self):
        """Execute the whole pipeline."""

        def _shutdown_cluster(signum, frame):
            logging.info("Shutdown urgently.")
            Master.shutdown()
            os._exit(0)

        try:
            signal.signal(signal.SIGINT, _shutdown_cluster)
            signal.signal(signal.SIGTERM, _shutdown_cluster)
            for step_name in self._steps:
                step_cfg = self.cfg.get(step_name)
                self.cfg.general["step_name"] = step_name
                ClassFactory().set_current_step(step_cfg)
                logger.info("Start pipeline step: [{}]".format(step_name))
                PipeStep().do()
        except Exception:
            logger.error("Failed to run pipeline.")
            logger.error(traceback.format_exc())
        try:
            Master.shutdown()
        except Exception:
            logger.error("Failed to shutdown dask cluster.")
            logger.error(traceback.format_exc())
