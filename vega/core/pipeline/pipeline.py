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

"""Pipeline that string up all PipeSteps."""

import os
import traceback
import logging
import signal
import json
import pandas as pd
from vega.common.user_config import UserConfig
from vega.common import FileOps, TaskOps, Status
from vega.core.scheduler import shutdown_cluster
from vega.common.general import General
from vega.report import ReportServer
from vega.common.message_server import MessageServer
from .pipe_step import PipeStep
from .conf import PipeStepConfig, PipelineConfig

logger = logging.getLogger(__name__)


class Pipeline(object):
    """Load configs and provide `run` method to start pipe steps.

    In this class, Pipeline will parse all pipe steps from the config data.
    Execute steps one by one and set glob configs with current step config.
    """

    def __init__(self):
        self.steps = []

    def run(self):
        """Execute the whole pipeline."""

        def _shutdown_cluster(signum, frame):
            logging.info("Shutdown urgently.")
            shutdown_cluster()
            os._exit(0)

        error_occured = False

        # start MessageServer
        MessageServer().run(ip=General.cluster.master_ip)
        General.message_port = MessageServer().port

        # start Report Server
        ReportServer().run()
        ReportServer().set_step_names(PipelineConfig.steps)

        try:
            signal.signal(signal.SIGINT, _shutdown_cluster)
            signal.signal(signal.SIGTERM, _shutdown_cluster)
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

                pipestep = PipeStep(name=step_name)
                self.steps.append(pipestep)
                pipestep.do()
        except Exception as e:
            logger.error(f"Failed to run pipeline, message: {e}")
            logger.error(traceback.format_exc())
            error_occured = True
            if "pipestep" in locals():
                pipestep.update_status(Status.error, str(e))

        shutdown_cluster()

        if not error_occured:
            self._show_pipeline_info()

    def _set_evaluator_config(self, step_cfg):
        if "evaluator" in step_cfg:
            if "host_evaluator" in step_cfg["evaluator"]:
                PipeStepConfig.evaluator.host_evaluator_enable = True
                PipeStepConfig.evaluator_enable = True
            if "device_evaluator" in step_cfg["evaluator"]:
                PipeStepConfig.evaluator.device_evaluator_enable = True
                PipeStepConfig.evaluator_enable = True

    def _interval_time(self, start, end):
        time_difference = end - start
        seconds = time_difference.seconds
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return "%d:%02d:%02d" % (hours + time_difference.days * 24, minutes, seconds)

    def _show_pipeline_info(self):
        logging.info("-" * 48)
        logging.info("  Pipeline end.")
        logging.info("")
        logging.info("  task id: {}".format(General.task.task_id))
        logging.info("  output folder: {}".format(TaskOps().local_output_path))
        logging.info("")
        self._show_step_time()
        logging.info("")
        self._show_report()
        logging.info("-" * 48)

    def _show_step_time(self):
        logging.info("  running time:")
        for step in self.steps:
            #  nas:  0:01:45  [2021-03-29 06:44:19.033835 - 2021-03-29 06:46:04.792390]
            step_time = self._interval_time(step.start_time, step.end_time)
            logging.info("  {:>16s}:  {}  [{} - {}]".format(step.name, step_time, step.start_time, step.end_time))

    def _show_report(self):
        performance_file = FileOps.join_path(
            TaskOps().local_output_path, self.steps[-1].name, "output.csv")
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
