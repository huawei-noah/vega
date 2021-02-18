# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Pipeline that string up all PipeSteps."""
import os
import traceback
import logging
import signal
from .pipe_step import PipeStep
from zeus.common.user_config import UserConfig
from vega.core.scheduler import shutdown_cluster
from .conf import PipeStepConfig, PipelineConfig
from zeus.common.general import General

logger = logging.getLogger(__name__)


class Pipeline(object):
    """Load configs and provide `run` method to start pipe steps.

    In this class, Pipeline will parse all pipe steps from the config data.
    Execute steps one by one and set glob configs with current step config.
    """

    def run(self):
        """Execute the whole pipeline."""

        def _shutdown_cluster(signum, frame):
            logging.info("Shutdown urgently.")
            shutdown_cluster()
            os._exit(0)

        try:
            signal.signal(signal.SIGINT, _shutdown_cluster)
            signal.signal(signal.SIGTERM, _shutdown_cluster)
            for step_name in PipelineConfig.steps:
                step_cfg = UserConfig().data.get(step_name)
                General.step_name = step_name
                PipeStepConfig.renew()
                PipeStepConfig.from_json(step_cfg, skip_check=False)
                self._set_evaluator_config(step_cfg)
                logger.info("Start pipeline step: [{}]".format(step_name))
                logger.debug("Pipe step config: {}".format(PipeStepConfig()))
                if PipeStepConfig.type == "NasPipeStep":
                    General._parallel = General.parallel_search
                if PipeStepConfig.type == "FullyTrainPipeStep":
                    General._parallel = General.parallel_fully_train
                PipeStep().do()
        except Exception:
            logger.error("Failed to run pipeline.")
            logger.error(traceback.format_exc())
        try:
            shutdown_cluster()
        except Exception:
            logger.error("Failed to shutdown dask cluster.")
            logger.error(traceback.format_exc())

    def _set_evaluator_config(self, step_cfg):
        if "evaluator" in step_cfg:
            if "gpu_evaluator" in step_cfg["evaluator"]:
                PipeStepConfig.evaluator.gpu_evaluator_enable = True
                PipeStepConfig.evaluator_enable = True
            if "davinci_mobile_evaluator" in step_cfg["evaluator"]:
                PipeStepConfig.evaluator.davinci_mobile_evaluator_enable = True
                PipeStepConfig.evaluator_enable = True
