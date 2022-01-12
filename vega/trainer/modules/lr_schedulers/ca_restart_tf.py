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

"""Cosine annealing restart lr scheduler."""
import math
import tensorflow as tf
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LR_SCHEDULER)
class CosineAnnealingRestartLR(object):
    """Cosine annealing learning rate with restarting.

    :param optimizer: original tf optimizer
    :type optimizer: tf.optimizer
    :param periods: periods of restarting point
    :type periods: list of int or float
    :param: restart_weights: restart weights of initial value
    :type restart_weights: list of float
    :param eta_min: minimized learning rate
    :type eta_min: float
    :param last_epoch: last epoch
    :type last_epoch: int
    :param warmup_epochs: warming up epochs
    :type warmup_epochs: int
    """

    def __init__(self, optimizer, periods, restart_weights=(1, ), eta_min=0, last_epoch=-1, warmup_epochs=None):
        """Initialize CosineAnnealingRestartLR."""
        self.optimizer = optimizer
        self.periods = list(map(float, periods))
        self.restart_weights = list(map(float, restart_weights))
        self.eta_min = eta_min
        if len(self.periods) != len(self.restart_weights):
            raise Exception("Periods length must be equal to restart weights")
        self.cumulative_period = [sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))]
        self.base_lr = optimizer.base_lr
        self.last_epoch = last_epoch
        self.milestones = self.cumulative_period[:-1]
        self.cumulative_period = [0.] + self.milestones
        self.current_lr = tf.cast(self.base_lr, tf.float32)
        self.warmup_epochs = warmup_epochs

    def _calc_lr(self, current_weight, nearest_restart, current_period, epoch):
        cosine = tf.math.cos(math.pi * (epoch - nearest_restart) / current_period)
        lr = self.eta_min + current_weight * 0.5 * (self.base_lr - self.eta_min) * (1 + cosine)
        return lr

    def step(self, epoch):
        """Set leanrning rate on global step."""
        epoch = tf.cast(epoch, tf.float32)
        current_weight = tf.compat.v1.train.piecewise_constant(epoch, self.milestones, self.restart_weights)
        nearest_restart = tf.compat.v1.train.piecewise_constant(epoch, self.milestones, self.cumulative_period)
        current_period = tf.compat.v1.train.piecewise_constant(epoch, self.milestones, self.periods)
        self.current_lr = self._calc_lr(current_weight, nearest_restart, current_period, epoch)
        if self.warmup_epochs is not None:
            warmup_ratio = (epoch - nearest_restart) / self.warmup_epochs
            warmup_lr = current_weight * self.base_lr * warmup_ratio
            self.current_lr = tf.cond((epoch - nearest_restart) < self.warmup_epochs,
                                      lambda: warmup_lr,
                                      lambda: self.current_lr)
        self.optimizer.set_lr(self.current_lr)

    def get_lr(self):
        """Get current learning rate."""
        return [self.current_lr]
