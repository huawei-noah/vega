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
from copy import deepcopy
import time
from .pipe_step import PipeStep
from .generator import Generator
from ..scheduler import create_master
from zeus.common import ClassFactory, ClassType
from ..pipeline.conf import PipeStepConfig
from zeus.report import Report
from zeus.common.general import General


@ClassFactory.register(ClassType.PIPE_STEP)
class NasPipeStep(PipeStep):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self):
        """Initialize."""
        super().__init__()
        self.generator = Generator.restore()
        if not self.generator:
            self.generator = Generator()
        Report.restore()
        self.master = create_master(update_func=self.generator.update)

    def do(self):
        """Do the main task in this pipe step."""
        logging.debug("NasPipeStep started...")
        while not self.generator.is_completed:
            res = self.generator.sample()
            if res:
                self._dispatch_trainer(res)
            else:
                time.sleep(0.2)
        self.master.join()
        logging.debug("Pareto_front values: %s", Report().pareto_front(General.step_name))
        Report().output_pareto_front(General.step_name)
        self.master.close_client()

    def _dispatch_trainer(self, samples):
        for (id_ele, desc) in samples:
            hps = deepcopy(desc)
            cls_trainer = ClassFactory.get_cls(ClassType.TRAINER)
            trainer = cls_trainer(id=id_ele, model_desc=desc, hps=hps)
            evaluator = self._get_evaluator(trainer)
            logging.info("submit trainer, id={}".format(id_ele))
            self.master.run(trainer, evaluator)
        if isinstance(samples, list) and len(samples) > 1:
            self.master.join()

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
