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
from copy import deepcopy
from .generator_mf import GeneratorMF
from zeus.common import ClassFactory, ClassType, UserConfig
from .nas_pipe_step import NasPipeStep
import os
import json
import torch
import numpy as np
from zeus.report import Report
from zeus.common.general import General
from ..scheduler import create_master

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class NasPipeStepMF(NasPipeStep):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self):
        """Initialize an instance of the NasPipeStepMF class."""
        super(NasPipeStep, self).__init__()
        self.generator = GeneratorMF.restore()
        if not self.generator:
            self.generator = GeneratorMF()
        Report.restore()
        self.master = create_master(update_func=self.generator.update)

    def _dispatch_trainer(self, gs):
        """Process a sampled instance."""
        id_ele, model_desc, epochs_ele = gs
        hps = deepcopy(model_desc)
        cls_trainer = ClassFactory.get_cls(ClassType.TRAINER)
        trainer = cls_trainer(id=id_ele, model_desc=model_desc, hps=hps)
        trainer.epochs = epochs_ele
        evaluator = self._get_evaluator(trainer)
        logger.info("submit trainer(id={})!".format(id_ele))
        self.master.run(trainer, evaluator)

        self.master.join()
