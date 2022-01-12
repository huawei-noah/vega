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

"""Benchmark Pipe Step defined in Pipeline."""
import logging
import os
import traceback

from vega.common import FileOps, Config, TaskOps
from vega.common import ClassFactory, ClassType
from vega.common.general import General
from vega.core.pipeline.conf import PipeStepConfig, PipelineConfig
from vega.evaluator.conf import EvaluatorConfig
from vega.report import ReportClient, ReportRecord, ReportServer
from vega.common import Status
from .pipe_step import PipeStep
from ..scheduler import create_master

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class BenchmarkPipeStep(PipeStep):
    """Run pipeStep which is the base components class that can be added in Pipeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do(self):
        """Start to run benchmark evaluator."""
        logger.info("BenchmarkPipeStep started...")
        records = self._get_current_step_records()
        if not records:
            logger.error("There is no model to evaluate.")
            return
        self.update_status(Status.running)
        self.master = create_master()
        for record in records:
            ReportClient().update(record.step_name, record.worker_id, desc=record.desc)
            self._evaluate_single_model(record)
        self.master.join()
        ReportServer().output_step_all_records(step_name=General.step_name)
        self.master.close()
        ReportServer().backup_output_path()
        self.update_status(Status.finished)

    def _get_current_step_records(self):
        step_name = General.step_name
        models_folder = PipeStepConfig.pipe_step.get("models_folder")
        cur_index = PipelineConfig.steps.index(step_name)
        if cur_index >= 1 or models_folder:
            if not models_folder:
                models_folder = FileOps.join_path(
                    TaskOps().local_output_path, PipelineConfig.steps[cur_index - 1])
            models_folder = models_folder.replace(
                "{local_base_path}", TaskOps().local_base_path)
            records = ReportServer().load_records_from_model_folder(models_folder)
        else:
            records = self._load_single_model_records()
        final_records = []
        for record in records:
            if not record.weights_file:
                logger.error("Model file is not existed, id={}".format(record.worker_id))
            else:
                record.step_name = General.step_name
                final_records.append(record)
        logging.debug("Records: {}".format(final_records))
        return final_records

    def _load_single_model_records(self):
        model_desc = PipeStepConfig.model.model_desc
        model_desc_file = PipeStepConfig.model.model_desc_file
        if model_desc_file:
            model_desc_file = model_desc_file.replace(
                "{local_base_path}", TaskOps().local_base_path)
            model_desc = Config(model_desc_file)
        if not model_desc:
            logger.error("Model desc or Model desc file is None.")
            return []
        model_file = PipeStepConfig.model.pretrained_model_file
        if not model_file:
            logger.error("Model file is None.")
            return []
        if not os.path.exists(model_file):
            logger.error("Model file is not existed.")
            return []
        return [ReportRecord().load_dict(dict(worker_id="1", desc=model_desc, weights_file=model_file))]

    def _evaluate_single_model(self, record):
        try:
            worker_info = {"step_name": record.step_name, "worker_id": record.worker_id}
            _record = dict(worker_id=record.worker_id, desc=record.desc, step_name=record.step_name)
            ReportClient().update(**_record)
            if EvaluatorConfig.host_evaluator_enable:
                cls_evaluator = ClassFactory.get_cls(ClassType.HOST_EVALUATOR, "HostEvaluator")
                evaluator = cls_evaluator(
                    worker_info=worker_info,
                    model_desc=record.desc,
                    weights_file=record.weights_file)
                self.master.run(evaluator)
            if EvaluatorConfig.device_evaluator_enable:
                cls_evaluator = ClassFactory.get_cls(
                    ClassType.DEVICE_EVALUATOR, "DeviceEvaluator")
                evaluator = cls_evaluator(
                    worker_info=worker_info,
                    model_desc=record.desc,
                    weights_file=record.weights_file)
                self.master.run(evaluator)
        except Exception as e:
            logger.error(f"Failed to evaluate model, worker info: {worker_info}, message: {e}")
            logger.debug(traceback.format_exc())
            return
