# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ModelStatistics callback defination."""
import logging
import torch
from .callbacks import Callback
from vega.core.metrics.pytorch import calc_model_flops_params
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class ModelStatistics(Callback):
    """Callback that log statistics about model after each epoch."""

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.input = None
        self.gflops = None
        self.kparams = None

    def after_train_step(self, batch_index, logs=None):
        """Be called after each batch of Training."""
        if self.input is None:
            input, target = logs['train_batch']
            self.input = torch.unsqueeze(input[0], 0)

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        self.model = self.trainer.model
        try:
            if self.gflops is None:
                flops_count, params_count = calc_model_flops_params(self.model,
                                                                    self.input)
                self.gflops, self.kparams = flops_count * 1600 * 1e-9, params_count * 1e-3
            summary_perfs = logs.get('summary_perfs', {})
            summary_perfs.update({'gflops': self.gflops, 'kparams': self.kparams,
                                  'epoch': epoch})
            logs.update({'summary_perfs': summary_perfs})
        except Exception as ex:
            logging.warning("model statics failed, ex=%s", ex)
