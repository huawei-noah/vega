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

"""Nas Pipe Step defined in Pipeline."""
import logging
import time
import os
import glob
import shutil
from vega.common import ClassFactory, ClassType
from vega.report import ReportServer
from vega.common.general import General
from vega.common import TaskOps, Status
from vega.trainer.conf import TrainerConfig
from .pipe_step import PipeStep
from .generator import Generator
from ..scheduler import create_master
from ..pipeline.conf import PipeStepConfig


@ClassFactory.register(ClassType.PIPE_STEP)
class SearchPipeStep(PipeStep):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        if not hasattr(self, "generator"):
            self.generator = Generator.restore()
        if not self.generator:
            self.generator = Generator()
        self.master = create_master(update_func=self.generator.update)
        self.user_trainer_config = TrainerConfig().to_dict()

    def do(self):
        """Do the main task in this pipe step."""
        super().do()
        logging.debug("SearchPipeStep started...")

        if hasattr(self.generator, "search_alg") and hasattr(self.generator.search_alg, "max_samples"):
            self.num_models = self.generator.search_alg.max_samples
            self.num_epochs = self.num_models * TrainerConfig.epochs
        self.update_status(Status.running)

        while not self.generator.is_completed:
            res = self.generator.sample()
            if res:
                self._dispatch_trainer(res)
            else:
                time.sleep(0.2)
        self.master.join()
        logging.debug("Pareto_front values: %s", ReportServer().pareto_front(General.step_name))
        ReportServer().output_pareto_front(General.step_name)
        self.master.close()
        if General.clean_worker_dir:
            self._clean_checkpoint()
        self.update_status(Status.finished)

    def _dispatch_trainer(self, samples):
        for (id, desc, hps) in samples:
            cls_trainer = ClassFactory.get_cls(ClassType.TRAINER, PipeStepConfig.trainer.type)
            TrainerConfig.from_dict(self.user_trainer_config)
            trainer = cls_trainer(id=id, model_desc=desc, hps=hps)
            evaluator = self._get_evaluator(trainer)
            logging.info("submit trainer, id={}".format(id))
            self.master.run(trainer, evaluator)

    def _get_evaluator(self, trainer):
        if not PipeStepConfig.evaluator_enable:
            return None
        try:
            cls_evaluator = ClassFactory.get_cls(ClassType.EVALUATOR)
        except Exception as e:
            logging.warning("Get evaluator failed:{}".format(str(e)))
            raise e
        if cls_evaluator is not None:
            evaluator = cls_evaluator({
                "step_name": General.step_name,
                "worker_id": trainer.worker_id})
            return evaluator
        else:
            return None

    def _clean_checkpoint(self):
        worker_parent_folder = os.path.abspath(
            os.path.join(TaskOps().get_local_worker_path(General.step_name, 1), ".."))
        patterns = [
            ".*.pkl", "*.pth", "model_*", "model.ckpt-*",
            "graph.*", "eval", "events*", "CKP-*", "checkpoint", ".*.log",
            "*.ckpt",  "*.caffemodel",
            "*.pbtxt", "*.bin", "kernel_meta", "*.prototxt",
        ]
        all_files = []
        worker_folders = glob.glob(worker_parent_folder + "/*")
        for worker_folder in worker_folders:
            for pattern in patterns:
                file_pattern = worker_folder + "/" + pattern
                all_files += glob.glob(file_pattern)
        if all_files:
            logging.info("Clean worker folder {}.".format(worker_parent_folder))
            for item in all_files:
                try:
                    if os.path.isfile(item):
                        os.remove(item)
                    elif os.path.isdir(item):
                        shutil.rmtree(item)
                except Exception:
                    logging.warn("Failed to remove {}".format(item))
