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
import datetime
import pandas as pd
import json
from .pipe_step import PipeStep
from zeus.common.user_config import UserConfig
from zeus.common.file_ops import FileOps
from zeus.common.task_ops import TaskOps
from vega.core.scheduler import shutdown_cluster
from zeus.common.general import General
from .conf import PipeStepConfig, PipelineConfig
from zeus.report import ReportServer

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

        steps_time = []
        error_occured = False

        try:
            signal.signal(signal.SIGINT, _shutdown_cluster)
            signal.signal(signal.SIGTERM, _shutdown_cluster)
            _ = ReportServer()
            for step_name in PipelineConfig.steps:
                step_cfg = UserConfig().data.get(step_name)
                General.step_name = step_name
                PipeStepConfig.renew()
                PipeStepConfig.from_dict(step_cfg, skip_check=False)
                self._set_evaluator_config(step_cfg)
                logging.info("-" * 48)
                logging.info("  Step: {}".format(step_name))
                logging.info("-" * 48)
                logger.debug("Pipe step config: {}".format(PipeStepConfig()))
                if PipeStepConfig.type == "SearchPipeStep":
                    General._parallel = General.parallel_search
                if PipeStepConfig.type == "TrainPipeStep":
                    General._parallel = General.parallel_fully_train

                start_time = datetime.datetime.now()
                PipeStep().do()
                end_time = datetime.datetime.now()
                steps_time.append([step_name, start_time, end_time, self._interval_time(start_time, end_time)])
        except Exception:
            logger.error("Failed to run pipeline.")
            logger.error(traceback.format_exc())
            error_occured = True
        try:
            shutdown_cluster()
        except Exception:
            logger.error("Failed to shutdown dask cluster.")
            logger.error(traceback.format_exc())

        if not error_occured:
            self._show_pipeline_info(steps_time, step_name)

    def _set_evaluator_config(self, step_cfg):
        if "evaluator" in step_cfg:
            if "host_evaluator" in step_cfg["evaluator"]:
                PipeStepConfig.evaluator.host_evaluator_enable = True
                PipeStepConfig.evaluator_enable = True
            if "device_evaluator" in step_cfg["evaluator"]:
                PipeStepConfig.evaluator.device_evaluator_enable = True
                PipeStepConfig.evaluator_enable = True

    def _interval_time(self, start, end):
        seconds = (end - start).seconds
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return "%d:%02d:%02d" % (hours, minutes, seconds)

    def _show_pipeline_info(self, steps_time, step_name):
        logging.info("-" * 48)
        logging.info("  Pipeline end.")
        logging.info("")
        logging.info("  task id: {}".format(General.task.task_id))
        logging.info("  output folder: {}".format(TaskOps().local_output_path))
        logging.info("")
        self._show_step_time(steps_time)
        logging.info("")
        self._show_report(step_name)
        logging.info("-" * 48)

    def _show_step_time(self, steps_time):
        logging.info("  running time:")
        for step in steps_time:
            logging.info("  {:>16s}:  {}  [{} - {}]".format(str(step[0]), step[3], step[1], step[2]))

    def _show_report(self, step_name):
        performance_file = FileOps.join_path(
            TaskOps().local_output_path, step_name, "output.csv")
        try:
            data = pd.read_csv(performance_file)
        except Exception:
            logging.info("  result file output.csv is not existed or empty")
            return
        if data.shape[1] < 2 or data.shape[0] == 0:
            logging.info("  result file output.csv is empty")
            return
        logging.info("  result:")
        data = json.loads(data.to_json())
        for key in data["worker_id"].keys():
            logging.info("  {:>3s}:  {}".format(str(data["worker_id"][key]), data["performance"][key]))
