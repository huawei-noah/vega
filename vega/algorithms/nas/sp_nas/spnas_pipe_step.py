# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""PipeStep for SpNAS."""

import logging
from vega.core.pipeline.nas_pipe_step import NasPipeStep
from vega.core.pipeline.generator import Generator
from vega.core.scheduler.master import Master
from vega.core.common.class_factory import ClassFactory, ClassType

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class SpNasPipeStep(NasPipeStep):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.master = Master()

    def do(self):
        """Do the main task in this pipe step."""
        logger.info("SpNasPipeStep started...")
        while not self.generator.is_completed:
            id, sample = self.generator.search_alg.search()
            cls_trainer = ClassFactory.get_cls('trainer')
            trainer = cls_trainer(sample, id)
            logging.info("submit trainer(id={})!".format(id))
            self.master.run(trainer)
            finished_trainer_info = self.master.pop_finished_worker()
            print(finished_trainer_info)
            self.update_generator(self.generator, finished_trainer_info)
        self.master.join()
        finished_trainer_info = self.master.pop_all_finished_train_worker()
        self.update_generator(self.generator, finished_trainer_info)
