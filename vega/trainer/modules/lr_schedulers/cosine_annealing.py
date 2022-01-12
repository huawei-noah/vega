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

"""Cosine annealing lr scheduler."""
import math
import tensorflow as tf
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LR_SCHEDULER)
class CosineAnnealingLR(object):
    """Cosine anealing learning rate with warm up.

    :param T_max: maximum number of iterations
    :type T_max: int
    :param eta_min: minimum learning
    :type eta_min: int
    :param last_epoch: index of last epoch
    :type last_epoch: float
    :param warmup: whether to warm up
    :type warmup: bool
    :param warmup_epochs: warmup epochs
    :type warmup_epochs: int
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup=False, warmup_epochs=5):
        """Init CosineAnnealingLR."""
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lr = optimizer.base_lr
        self.current_lr = self.base_lr
        self.warmup = warmup
        self.warmup_epochs = warmup_epochs
        self.optimizer = optimizer

    def _calc_normal_lr(self):
        if self.last_epoch == 0:
            self.current_lr = self.base_lr
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            self.current_lr = self.current_lr + (self.base_lr - self.eta_min) * \
                (1 - math.cos(math.pi / self.T_max)) / 2
        else:
            self.current_lr = (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / \
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * \
                (self.current_lr - self.eta_min) + self.eta_min

    def _calc_closed_form_lr(self):
        self.current_lr = self.eta_min + (self.base_lr - self.eta_min) * \
            (1 + tf.math.cos(tf.constant(math.pi) * self.last_epoch / self.T_max)) / 2
        self.current_lr = tf.cast(self.current_lr, tf.float32)

    def step(self, epoch):
        """Obtain the learning rate on global steps."""
        epoch = tf.cast(epoch, tf.float32)
        self.last_epoch = epoch
        if hasattr(self, "_calc_closed_form_lr"):
            self._calc_closed_form_lr()
        else:
            self._calc_normal_lr()
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
