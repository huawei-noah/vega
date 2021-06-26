# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
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
