# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of segmentation task."""
import tensorflow as tf
from vega.common import ClassFactory, ClassType
from vega.metrics.tensorflow.metrics import MetricBase


@ClassFactory.register(ClassType.METRIC)
class IoUMetric(MetricBase):
    """Calculate IoU between output and target."""

    __metric_name__ = 'IoUMetric'

    def __init__(self, num_class):
        self.num_classes = num_class

    def __call__(self, output, target):
        """Calculate IoU.

        :param output: output of segmentation network
        :param target: ground truth from dataset
        :return: IoU value
        """
        output = tf.cast(tf.argmax(output, axis=1), tf.int32)
        weights = tf.to_float(tf.less(target, self.num_classes))
        target = tf.where(tf.less(target, self.num_classes), target, tf.zeros_like(target))
        iou_value = {'IoUMetric': tf.compat.v1.metrics.mean_iou(target, output, self.num_classes, weights=weights)}
        return iou_value
