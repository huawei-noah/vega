# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Nas Pipe Step defined in Pipeline."""
import logging
import traceback
from .pipe_step import PipeStep
from .generator import Generator
from ..scheduler.master import Master
from ..common.class_factory import ClassFactory, ClassType

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class NasPipeStep(PipeStep):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.master = Master()

    def do(self):
        """Do the main task in this pipe step."""
        logger.info("NasPipeStep started...")
        while not self.generator.is_completed:
            id, model = self.generator.sample()
            if isinstance(id, list) and isinstance(model, list):
                for id_ele, model_ele in zip(id, model):
                    cls_trainer = ClassFactory.get_cls('trainer')
                    trainer = cls_trainer(model_ele, id_ele)
                    logger.info("submit trainer(id={})!".format(id_ele))
                    self.master.run(trainer)
                self.master.join()
            elif id is not None and model is not None:
                cls_trainer = ClassFactory.get_cls('trainer')
                trainer = cls_trainer(model, id)
                logger.info("submit trainer(id={})!".format(id))
                self.master.run(trainer)
            finished_trainer_info = self.master.pop_finished_worker()
            self.update_generator(self.generator, finished_trainer_info)
        self.master.join()
        finished_trainer_info = self.master.pop_all_finished_train_worker()
        self.update_generator(self.generator, finished_trainer_info)
        self._backup_output_path()
        self.master.close_client()

    def update_generator(self, generator, worker_info):
        """Get finished worker's info, and use it to update target `generator`.

        Will get the finished worker's working dir, and then call the function
        `generator.update(worker_result_path)`.
        :param Generator generator: The target `generator` need to update.
        :param worker_info: `worker_info` is the finished worker's info, usually
            a dict or list of dict include `step_name` and `worker_id`.
        :type worker_info: dict or list of dict.

        """
        if worker_info is None:
            return
        if not isinstance(worker_info, list):
            worker_info = [worker_info]
        for one_info in worker_info:
            step_name = one_info["step_name"]
            worker_id = one_info["worker_id"]
            logger.info("update generator, step name: {}, worker id: {}".format(step_name, worker_id))
            try:
                worker_result_path = self.task.get_local_worker_path(step_name, worker_id)
                generator.update(worker_result_path)
            except Exception:
                logger.error("Failed to upgrade generator, step_name={}, worker_id={}.".format(
                    step_name, worker_id))
                logger.error(traceback.format_exc())
