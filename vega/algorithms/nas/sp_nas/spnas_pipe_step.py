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
from vega.core.pipeline.pipe_step import PipeStep
from vega.core.pipeline.generator import Generator
from vega.core.scheduler import create_master
from zeus.common import ClassFactory, ClassType

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class SpNasPipeStep(PipeStep):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self):
        """Initialize SpNasPipeStep."""
        super().__init__()
        self.generator = Generator()
        self.master = create_master(update_func=self.generator.search_alg.update)

    def do(self):
        """Do the main task in this pipe step."""
        logger.info("SpNasPipeStep started")
        while not self.generator.is_completed:
            id, spnas_sample = self.generator.search_alg.search()
            cls_trainer = ClassFactory.get_cls('trainer')
            trainer = cls_trainer(spnas_sample=spnas_sample, id=id)
            logging.info("submit trainer(id={})!".format(id))
            self.master.run(trainer)
        self.master.join()


@ClassFactory.register(ClassType.PIPE_STEP)
class SpNasFullyTrain(PipeStep):
    """PipeStep is the base components class that can be added in Pipeline."""

    def do(self):
        """Do the main task in this pipe step."""
        cls_trainer = ClassFactory.get_cls('trainer')
        trainer = cls_trainer().train_process()
