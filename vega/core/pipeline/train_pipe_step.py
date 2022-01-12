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

"""Fully Train PipeStep that used in Pipeline."""

import os
import logging
import vega
from vega.common.general import General
from vega.common.class_factory import ClassFactory, ClassType
from vega.common import FileOps, TaskOps, Status
from vega.report import ReportServer, ReportRecord, ReportClient
from vega.core.scheduler import create_master
from vega.core.pipeline.conf import PipeStepConfig, PipelineConfig
from vega.trainer.conf import TrainerConfig
from .pipe_step import PipeStep

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class TrainPipeStep(PipeStep):
    """TrainPipeStep is the implementation class of PipeStep.

    Fully train is the last pipe step in pipeline, we provide horovrd or local trainer
    for user to choose.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("init TrainPipeStep...")

    def do(self):
        """Start to run fully train with horovod or local trainer."""
        super().do()
        logger.info("TrainPipeStep started...")
        records = self._get_current_step_records()
        logger.debug("load pipestep records: {}".format(records))
        self.num_models = len(records)
        self.num_epochs = self.num_models * TrainerConfig.epochs
        self.update_status(Status.running)
        self.master = create_master()
        self._train_multi_models(records)
        self.master.join()
        ReportServer().output_step_all_records(step_name=self.task.step_name)
        self.master.close()
        ReportServer().backup_output_path()
        self.update_status(Status.finished)

    def _get_current_step_records(self):
        step_name = self.task.step_name
        models_folder = PipeStepConfig.pipe_step.get("models_folder")
        models_folder = models_folder or PipeStepConfig.pipe_step.get("hps_folder")
        cur_index = PipelineConfig.steps.index(step_name)
        if cur_index >= 1 or models_folder:
            if not models_folder:
                models_folder = FileOps.join_path(
                    TaskOps().local_output_path, PipelineConfig.steps[cur_index - 1])
            models_folder = models_folder.replace(
                "{local_base_path}", TaskOps().local_base_path)
            records = ReportServer().load_records_from_model_folder(models_folder)
        else:
            records = [ReportRecord(step_name, 0)]
        logging.debug("Records: {}".format(records))
        for record in records:
            record.step_name = step_name
        return records

    def _train_multi_models(self, records):
        for record in records:
            weights_file = record.weights_file if PipeStepConfig.pipe_step.get("load_weights", True) else None
            trainer = self._build_trainer(
                model_desc=record.desc, hps=record.hps, model_id=record.worker_id, weights_file=weights_file)
            self.train_model(trainer)

    def _build_trainer(self, model_desc=None, hps=None, model_id=None, weights_file=None):
        cls_trainer = ClassFactory.get_cls(ClassType.TRAINER, PipeStepConfig.trainer.type)
        step_name = self.task.step_name
        if model_desc is not None:
            sample = dict(worker_id=model_id, desc=model_desc, step_name=step_name, weights_file=weights_file)
            record = ReportRecord().load_dict(sample)
            logging.debug("update record=%s", str(record))
            trainer = cls_trainer(model_desc=model_desc, hps=hps, id=model_id, pretrained_model_file=weights_file)
        else:
            trainer = cls_trainer(None, 0, hps=hps)
            record = ReportRecord(trainer.step_name, trainer.worker_id, desc=trainer.model_desc, hps=hps,
                                  weights_file=weights_file)
        ReportClient().update(**record.to_dict())
        # resume training
        if vega.is_torch_backend() and General._resume:
            trainer.load_checkpoint = True
            trainer._resume_training = True
        return trainer

    def train_model(self, trainer):
        """Train model."""
        evaluator = self._get_evaluator(trainer.worker_id)
        self.master.run(trainer, evaluator)

    def _get_evaluator(self, worker_id):
        if not PipeStepConfig.evaluator_enable:
            return None
        cls_evaluator = ClassFactory.get_cls('evaluator', "Evaluator")
        evaluator = cls_evaluator({"step_name": self.task.step_name, "worker_id": worker_id})
        return evaluator
