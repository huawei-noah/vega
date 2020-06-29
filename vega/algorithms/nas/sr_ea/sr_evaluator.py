# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""SRGpuEvaluator used to do evaluate process on gpu."""
import os
import time
import math
import logging
import torch
import numpy as np
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.evaluator.gpu_evaluator import GpuEvaluator
from vega.core.metrics.pytorch import Metrics


logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.GPU_EVALUATOR)
class SrGpuEvaluator(GpuEvaluator):
    """Evaluator is a gpu evaluator.

    :param args: arguments from user and default config file
    :type args: dict or Config, default to None
    :param train_data: training dataset
    :type train_data: torch dataset, default to None
    :param valid_data: validate dataset
    :type valid_data: torch dataset, default to None
    :param worker_info: the dict worker info of workers that finished train.
    :type worker_info: dict or None.

    """

    def __init__(self, worker_info=None, model=None, hps=None, **kwargs):
        """Init GpuEvaluator."""
        super(SrGpuEvaluator, self).__init__(worker_info, model, hps, **kwargs)

    def valid(self, loader):
        """Validate one step of model.

        :param loader: validation dataloader
        """
        metrics = Metrics(self.cfg.metric)
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                img_lr, img_hr = batch["LR"].cuda() / 255.0, batch["HR"].cuda() / 255.0
                image_sr = self.model(img_lr)
                metrics(image_sr, img_hr)  # round images gives lower results
        performance = metrics.results
        logging.info('Valid metric: {}'.format(performance))
        return performance
