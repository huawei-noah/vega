# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Multi step warm up lr scheduler."""
import tensorflow as tf
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LR_SCHEDULER)
class StepLR(object):
    """Step learning rate.

    :param milestones: list of decay epochs
    :type milestones: list of int
    :param decay_rates: list of decay rates
    :type decay_rates: list of float
    :param warmup: whether to warm up
    :type warmup: bool
    :param epoch_steps: steps in one epoch
    :type epoch_steps: int
    """

    def __init__(self, optimizer, step_size, gamma):
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.base_lr
        self.current_lr = optimizer.base_lr
        self.optimizer = optimizer

    def step(self, epoch):
        """Obtain the learning rate on global steps."""
        epoch = tf.cast(epoch, tf.float32)
        self.current_lr = self.base_lr * self.gamma ** (epoch // self.step_size)
        self.optimizer.set_lr(self.current_lr)

    def get_lr(self):
        """Get current learning rate."""
        return [self.current_lr]
