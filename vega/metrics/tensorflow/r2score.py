# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of Regression task."""
import tensorflow as tf
from vega.metrics.tensorflow.metrics import MetricBase
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.METRIC, alias='r2score')
class R2Score(MetricBase):
    """Calculate R2 Score between output and target."""

    __metric_name__ = 'r2score'

    def __init__(self):
        pass

    def __call__(self, output, target, *args, **kwargs):
        """Forward and calculate r2 score."""
        residual = tf.reduce_sum(tf.square(tf.subtract(target, output)))
        total = tf.reduce_sum(tf.square(tf.subtract(target, tf.reduce_mean(target))))
        r2 = tf.subtract(1.0, tf.div(residual, total))
        r2score = {self.__metric_name__: tf.compat.v1.metrics.mean(r2)}
        return r2score
