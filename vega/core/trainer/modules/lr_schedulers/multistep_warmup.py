# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Multi step warm up lr scheduler."""
import importlib
import tensorflow as tf
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.LR_SCHEDULER)
class MultiStepLrWarmUp(object):
    """Multiple Step learning rate with warm up.

    :param base_lr: base learning rate
    :type base_lr: float
    :param milestones: list of decay epochs
    :type milestones: list of int
    :param decay_rates: list of decay rates
    :type decay_rates: list of float
    :param warmup: whether to warm up
    :type warmup: bool
    :param epoch_steps: steps in one epoch
    :type epoch_steps: int
    """

    def __init__(self, base_lr, milestones, decay_rates, warmup=True, warmup_epochs=5):
        self.milestones = list(map(float, milestones))
        self.milestone_lrs = [base_lr * decay for decay in decay_rates]
        self.base_lr = base_lr
        self.warmup = warmup
        self.warmup_epochs = tf.cast(warmup_epochs, tf.float32)
        self.current_lr = base_lr

    def step(self, epoch):
        """Obtain the learning rate on global steps."""
        epoch = tf.cast(epoch, tf.float32)
        self.current_lr = tf.train.piecewise_constant(epoch, self.milestones, self.milestone_lrs)
        if self.warmup:
            warmup_ratio = epoch / self.warmup_epochs
            warmup_lr = self.base_lr * warmup_ratio
            self.current_lr = tf.cond(epoch < self.warmup_epochs,
                                      lambda: warmup_lr,
                                      lambda: self.current_lr)

    def get_lr(self):
        """Get current learning rate."""
        return [self.current_lr]
