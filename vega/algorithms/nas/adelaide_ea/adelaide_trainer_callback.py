# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The trainer program for Adelaide_EA."""
import logging

import vega
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.metrics import calc_model_flops_params
from vega.core.trainer.callbacks import Callback

if vega.is_torch_backend():
    import torch
elif vega.is_tf_backend():
    import tensorflow as tf

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class AdelaideEATrainerCallback(Callback):
    """Construct the trainer of Adelaide-EA."""

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.config = self.trainer.config
        if vega.is_torch_backend():
            count_input = torch.FloatTensor(1, 3, 192, 192).cuda()
        elif vega.is_tf_backend():
            tf.reset_default_graph()
            count_input = tf.random_uniform([1, 192, 192, 3], dtype=tf.float32)
        flops_count, params_count = calc_model_flops_params(self.trainer.model, count_input)
        self.flops_count, self.params_count = flops_count * 1e-9, params_count * 1e-3
        logger.info("Flops: {:.2f} G, Params: {:.1f} K".format(self.flops_count, self.params_count))
        if self.flops_count > self.config.flops_limit:
            logger.info("Flop too large!")
            self.trainer.skip_train = True

    def after_epoch(self, epoch, logs=None):
        """Update gflops and kparams."""
        summary_perfs = logs.get('summary_perfs', {})
        summary_perfs.update({'gflops': self.flops_count, 'kparams': self.params_count})
        logs.update({'summary_perfs': summary_perfs})

    def make_batch(self, batch):
        """Make batch for each training step."""
        input = batch["data"]
        target = batch["mask"]
        if self.config.cuda:
            input = input.cuda()
            target = target.cuda()
        return input, target
