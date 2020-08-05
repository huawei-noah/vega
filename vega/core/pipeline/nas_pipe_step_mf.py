# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Nas Pipe Step (Multi-Fidelity) defined in Pipeline."""
import logging
import traceback
from .pipe_step import PipeStep
from .generator import Generator
from ..scheduler.master import Master
from ..common.class_factory import ClassFactory, ClassType

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class NasPipeStepMF(NasPipeStep):
    """PipeStep is the base components class that can be added in Pipeline."""

    def do(self):
        """Do the main task in this pipe step."""
        logger.info("NasPipeStep started...")
        while not self.generator.is_completed:
            id, model, epochs = self.generator.sample()
            if isinstance(id, list) and isinstance(model, list):
                for id_ele, model_ele, epochs_ele in zip(id, model, epochs):
                    cls_trainer = ClassFactory.get_cls('trainer')
                    trainer = cls_trainer(model_ele, id_ele)
                    trainer.epochs = epochs_ele
                    logger.info("submit trainer(id={})!".format(id_ele))
                    self.master.run(trainer)
                self.master.join()
            elif id is not None and model is not None:
                cls_trainer = ClassFactory.get_cls('trainer')
                trainer = cls_trainer(model, id)
                trainer.epochs = epochs
                logger.info("submit trainer(id={})!".format(id))
                self.master.run(trainer)
            finished_trainer_info = self.master.pop_finished_worker()
            self.update_generator(self.generator, finished_trainer_info)
        self.master.join()
        finished_trainer_info = self.master.pop_all_finished_train_worker()
        self.update_generator(self.generator, finished_trainer_info)
        self._backup_output_path()
        self.master.close_client()