# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined faster rcnn detector."""
import tensorflow as tf
from tensorflow.contrib import opt as tf_opt
from object_detection.utils import learning_schedules


class TFOptimizer(object):
    """TFOptimizer."""

    def __init__(self, desc):
        """Init TFOptimizer.

        :param desc: config dict
        """
        self.type = desc.type
        self.lr = desc.lr
        self.momentum = desc.momentum
        self.weight_decay = desc.weight_decay if hasattr(
            desc, 'weight_decay') else 0
        self.epsilon = desc.epsilon if hasattr(desc, 'epsilon') else 0
        self.use_moving_average = desc.use_moving_average if hasattr(
            desc, 'use_moving_average') else True
        self.moving_average_decay = desc.moving_average_decay if hasattr(
            desc, 'moving_average_decay') else 0.9999
        self.warmup = desc.warmup if hasattr(desc, 'warmup') else False
        self.optimizer = None
        self.summary_vars = []

    def _create_learning_rate(self, learning_rate_config, global_step=None):
        """Create optimizer learning rate based on config.

        Args:
            learning_rate_config: A LearningRate proto message.
            global_step: A variable representing the current step.
            If None, defaults to tf.train.get_or_create_global_step()
        Returns:
            A learning rate.
        Raises:
            ValueError: when using an unsupported input data type.
        """
        if global_step is None:
            global_step = tf.train.get_or_create_global_step()
        learning_rate = None
        learning_rate_type = learning_rate_config.type
        if learning_rate_type == 'constant_learning_rate':
            config = learning_rate_config.constant_learning_rate
            learning_rate = tf.constant(config.learning_rate, dtype=tf.float32,
                                        name='learning_rate')

        if learning_rate_type == 'exponential_decay_learning_rate':
            config = learning_rate_config.exponential_decay_learning_rate
            learning_rate = learning_schedules.exponential_decay_with_burnin(
                global_step,
                config.initial_learning_rate,
                config.decay_steps,
                config.decay_factor,
                burnin_learning_rate=config.burnin_learning_rate,
                burnin_steps=config.burnin_steps,
                min_learning_rate=config.min_learning_rate,
                staircase=config.staircase)

        if learning_rate_type == 'manual_step_learning_rate':
            config = learning_rate_config
            if not config.schedule:
                raise ValueError('Empty learning rate schedule.')
            learning_rate_step_boundaries = [x['step'] for x in config.schedule]
            learning_rate_sequence = [config.initial_learning_rate]
            learning_rate_sequence += [x['learning_rate']
                                       for x in config.schedule]
            learning_rate = learning_schedules.manual_stepping(
                global_step, learning_rate_step_boundaries,
                learning_rate_sequence, self.warmup)

        if learning_rate_type == 'cosine_decay_learning_rate':
            config = learning_rate_config.cosine_decay_learning_rate
            learning_rate = learning_schedules.cosine_decay_with_warmup(
                global_step,
                config.learning_rate_base,
                config.total_steps,
                config.warmup_learning_rate,
                config.warmup_steps,
                config.hold_base_rate_steps)

        if learning_rate is None:
            raise ValueError('Learning_rate %s not supported.' %
                             learning_rate_type)

        return learning_rate

    def get_real_optimizer(self, global_step=None):
        """Get real optimizer for faster-rcnn."""
        if self.optimizer:
            return self.optimizer, self.summary_vars
        else:
            if self.type == 'RMSPropOptimizer':
                learning_rate = self._create_learning_rate(self.lr,
                                                           global_step=global_step)
                self.summary_vars.append(learning_rate)
                self.optimizer = tf.train.RMSPropOptimizer(
                    learning_rate,
                    decay=self.weight_decay,
                    momentum=self.momentum,
                    epsilon=self.epsilon)

            if self.type == 'MomentumOptimizer':
                learning_rate = self._create_learning_rate(self.lr,
                                                           global_step=global_step)
                self.summary_vars.append(learning_rate)
                self.optimizer = tf.train.MomentumOptimizer(
                    learning_rate,
                    momentum=self.momentum)

            if self.type == 'AdamOptimizer':
                learning_rate = self._create_learning_rate(self.lr,
                                                           global_step=global_step)
                self.summary_vars.append(learning_rate)
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate, epsilon=self.epsilon)

            if self.optimizer is None:
                raise ValueError('Optimizer %s not supported.' % self.type)

            if self.use_moving_average:
                self.optimizer = tf_opt.MovingAverageOptimizer(
                    self.optimizer, average_decay=self.moving_average_decay)

            return self.optimizer, self.summary_vars
