# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of classifier task."""
import tensorflow as tf
from vega.core.metrics.tensorflow.metrics import MetricBase
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.METRIC)
class accuracy(MetricBase):
    """Calculate classification accuracy between output and target."""

    __metric_name__ = 'accuracy'

    def __init__(self, topk=(1,)):
        """Init accuracy metric."""
        self.topk = topk

    def __call__(self, output, target, *args, **kwargs):
        """Forward and calculate accuracy."""
        top_accuracy = {}
        is_one = True if len(self.topk) == 1 else False
        for k in self.topk:
            key = self.__metric_name__ if is_one else 'top{}_accuracy'.format(k)
            in_top_k = tf.cast(tf.nn.in_top_k(output, target, k), tf.float32)
            top_k_accuracy = tf.metrics.mean(in_top_k)
            top_accuracy[key] = top_k_accuracy
        return top_accuracy
