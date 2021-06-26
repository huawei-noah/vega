# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of Regression task."""
from vega.metrics.pytorch.metrics import MetricBase
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.METRIC)
class RMSE(MetricBase):
    """Calculate RMSE between output and target."""

    __metric_name__ = 'RMSE'

    def __call__(self, output, target, *args, **kwargs):
        """Perform top k accuracy.

        :param output: output of classification network
        :param target: ground truth from dataset
        :return: pfm
        """
        import tensorflow as tf
        mean, std, label = target
        label = tf.cast(label, tf.float32)
        std = tf.cast(std, tf.float32)
        mean = tf.cast(mean, tf.float32)
        pred = output * std + mean
        label = label * std + mean
        rmse = tf.sqrt(tf.losses.mean_squared_error(pred, label))
        return {'RMSE': tf.compat.v1.metrics.mean(rmse)}

    @property
    def objective(self):
        """Define reward mode, default is max."""
        return 'MIN'


@ClassFactory.register(ClassType.METRIC)
class MSE(MetricBase):
    """Calculate Mse accuracy between output and target."""

    __metric_name__ = 'MSE'

    def __call__(self, output, target, *args, **kwargs):
        """Perform top k accuracy.

        :param output: output of classification network
        :param target: ground truth from dataset
        :return: pfm
        """
        import tensorflow as tf
        mean, std, label = target
        label = tf.cast(label, tf.float32)
        std = tf.cast(std, tf.float32)
        mean = tf.cast(mean, tf.float32)
        pred = output * std + mean
        label = label * std + mean
        mse = tf.losses.mean_squared_error(pred, label)
        return {'MSE': tf.compat.v1.metrics.mean(mse)}

    @property
    def objective(self):
        """Define reward mode, default is max."""
        return 'MIN'
