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
import zeus
from .callback import Callback
from zeus.metrics import calc_model_flops_params, calc_forward_latency
from zeus.common import ClassFactory, ClassType

if zeus.is_torch_backend():
    import torch


@ClassFactory.register(ClassType.CALLBACK)
class ModelStatistics(Callback):
    """Callback that log statistics about model after each epoch."""

    def __init__(self):
        """Initialize ModelStatistics callback."""
        super(Callback, self).__init__()
        self.priority = 220

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.input = None
        self.flops = None
        self.params = None
        self.latency = None
        self.calc_params_each_epoch = self.trainer.config.calc_params_each_epoch
        if zeus.is_tf_backend():
            import tensorflow as tf
            datasets = self.trainer.valid_input_fn()
            data_iter = tf.compat.v1.data.make_one_shot_iterator(datasets)
            # data_iter = self.trainer.valid_input_fn().make_one_shot_iterator()
            input_data, _ = data_iter.get_next()
            self.input = input_data[:1]

    def after_train_step(self, batch_index, logs=None):
        """Be called after each batch of Training."""
        try:
            if self.input is None:
                input, target = logs['train_batch']
                self.input = torch.unsqueeze(input[0], 0)
        except Exception as ex:
            logging.warning("model statics failed, ex=%s", ex)

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        if self.calc_params_each_epoch:
            self.update_flops_params(epoch=epoch, logs=logs)

    def after_train(self, logs=None):
        """Be called after train."""
        if not self.calc_params_each_epoch:
            self.update_flops_params(logs=logs)

    def update_flops_params(self, epoch=None, logs=None):
        """Calculate flops and params."""
        self.model = self.trainer.model
        try:
            if self.flops is None:
                flops_count, params_count = calc_model_flops_params(self.model,
                                                                    self.input)
                self.flops, self.params = flops_count * 1e-9, params_count * 1e-3
            if self.latency is None:
                sess_config = self.trainer._init_session_config() if zeus.is_tf_backend() else None
                self.latency = calc_forward_latency(self.model, self.input, sess_config) * 1000
            summary_perfs = logs.get('summary_perfs', {})
            if epoch:
                summary_perfs.update({'flops': self.flops, 'params': self.params,
                                      'latency': self.latency, 'epoch': epoch})
            else:
                summary_perfs.update({'flops': self.flops, 'params': self.params,
                                      'latency': self.latency})
            logs.update({'summary_perfs': summary_perfs})
        except Exception as ex:
            logging.warning("model statics failed, ex=%s", ex)
