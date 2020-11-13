# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Benchmark Pipe Step defined in Pipeline."""
import logging
import os
import traceback

from zeus.common import FileOps, Config
from zeus.common import ClassFactory, ClassType
from zeus.common.general import General
from zeus.common.task_ops import TaskOps
from vega.core.pipeline.conf import PipeStepConfig, PipelineConfig
from zeus.evaluator.conf import EvaluatorConfig
from zeus.report import Report, ReportRecord
from .pipe_step import PipeStep
from ..scheduler import create_master

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class BenchmarkPipeStep(PipeStep):
    """Run pipeStep which is the base components class that can be added in Pipeline."""

    def __init__(self):
        super().__init__()

    def do(self):
        """Start to run benchmark evaluator."""
        logger.info("BenchmarkPipeStep started...")
        records = self._get_current_step_records()
        if not records:
            logger.error("There is no model to evaluate.")
            return
        self.master = create_master()
        for record in records:
            _record = ReportRecord(worker_id=record.worker_id, desc=record.desc, step_name=record.step_name)
            Report().broadcast(_record)
            self._evaluate_single_model(record)
        self.master.join()
        for record in records:
            Report().update_report({"step_name": record.step_name, "worker_id": record.worker_id})
        Report().output_step_all_records(
            step_name=General.step_name,
            weights_file=False,
            performance=True)
        self.master.close_client()
        Report().backup_output_path()

    def _get_current_step_records(self):
        step_name = General.step_name
        models_folder = PipeStepConfig.pipe_step.get("models_folder")
        cur_index = PipelineConfig.steps.index(step_name)
        if cur_index >= 1 or models_folder:
            # records = Report().get_step_records(PipelineConfig.steps[cur_index - 1])
            if not models_folder:
                models_folder = FileOps.join_path(
                    TaskOps().local_output_path, PipelineConfig.steps[cur_index - 1])
            models_folder = models_folder.replace(
                "{local_base_path}", TaskOps().local_base_path)
            records = Report().load_records_from_model_folder(models_folder)
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
            _init_record = ReportRecord().load_dict(_record)
            Report().broadcast(_init_record)
            if EvaluatorConfig.gpu_evaluator_enable:
                cls_evaluator = ClassFactory.get_cls(ClassType.GPU_EVALUATOR, "GpuEvaluator")
                evaluator = cls_evaluator(
                    worker_info=worker_info,
                    model_desc=record.desc,
                    weights_file=record.weights_file)
                self.master.run(evaluator)
            if EvaluatorConfig.davinci_mobile_evaluator_enable:
                cls_evaluator = ClassFactory.get_cls(
                    ClassType.DAVINCI_MOBILE_EVALUATOR, "DavinciMobileEvaluator")
                evaluator = cls_evaluator(
                    worker_info=worker_info,
                    model_desc=record.desc,
                    weights_file=record.weights_file)
                self.master.run(evaluator)
        except Exception:
            logger.error("Failed to evaluate model, worker info={}".format(worker_info))
            logger.error(traceback.format_exc())
            return
