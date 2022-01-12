# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi step warm up lr scheduler."""

import tensorflow as tf
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LR_SCHEDULER)
class MultiStepLR(object):
    """Multiple Step learning rate with warm up.

    :param milestones: list of decay epochs
    :type milestones: list of int
    :param decay_rates: list of decay rates
    :type decay_rates: list of float
    :param warmup: whether to warm up
    :type warmup: bool
    :param epoch_steps: steps in one epoch
    :type epoch_steps: int
    """

    def __init__(self, optimizer, milestones, gamma, warmup=False, warmup_epochs=5):
        self.optimizer = optimizer
        self.milestones = list(map(float, milestones))
        base_lr = self.optimizer.base_lr
        self.milestone_lrs = [base_lr * gamma ** i for i in range(len(self.milestones) + 1)]
        self.base_lr = self.optimizer.base_lr
        self.warmup = warmup
        self.warmup_epochs = tf.cast(warmup_epochs, tf.float32)
        self.current_lr = tf.cast(self.base_lr, tf.float32)

    def step(self, epoch):
        """Obtain the learning rate on global steps."""
        epoch = tf.cast(epoch, tf.float32)
        self.current_lr = tf.compat.v1.train.piecewise_constant(epoch, self.milestones, self.milestone_lrs)
        self.current_lr = tf.cast(self.current_lr, tf.float32)
        if self.warmup:
            warmup_ratio = epoch / self.warmup_epochs
            warmup_lr = self.base_lr * warmup_ratio
            self.current_lr = tf.cond(epoch < self.warmup_epochs,
                                      lambda: warmup_lr,
                                      lambda: self.current_lr)
        self.optimizer.set_lr(self.current_lr)

    def get_lr(self):
        """Get current learning rate."""
        return [self.current_lr]
