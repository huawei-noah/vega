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
"""Charbonnier Loss class."""
import functools
import tensorflow as tf
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LOSS)
class CharbonnierLoss(object):
    """Charbonnier Loss for TensorFlow."""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def __call__(self, logits, labels):
        """Loss forward function."""
        sqrt_loss = tf.sqrt(tf.square(tf.subtract(logits, labels)) + self.eps)
        if self.reduction == 'mean':
            loss = tf.reduce_mean(sqrt_loss)
        elif self.reduction == 'sum':
            loss = tf.reduce_sum(sqrt_loss)
        loss = self.loss_weight * loss
        return loss
