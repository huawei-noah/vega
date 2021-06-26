# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""FocalLoss for unbalanced data."""
from vega.modules.module import Module
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.LOSS)
class ForecastLoss(Module):
    """Forecast Loss for St data."""

    def __init__(self, epsilon=1e-4):
        super(ForecastLoss, self).__init__()
        self.epsilon = epsilon

    def call(self, y_pred, y_true):
        """Compute loss.

        :param inputs: predict data.
        :param targets: true data.
        :return:
        """
        import tensorflow as tf
        y_true = tf.cast(y_true[2], tf.float32)
        mae_loss = tf.reduce_sum(tf.losses.absolute_difference(y_true, y_pred))
        mse_loss = tf.nn.l2_loss(y_pred - y_true)
        return mae_loss + self.epsilon * mse_loss
