# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of super resolution task."""
import tensorflow as tf
from zeus.common import ClassFactory, ClassType
from zeus.metrics.tensorflow.metrics import MetricBase


@ClassFactory.register(ClassType.METRIC)
class SRMetric(MetricBase):
    """Calculate IoU between output and target."""

    __metric_name__ = 'SRMetric'

    def __init__(self, method='psnr', to_y=True, scale=2, max_rgb=1):
        self.method = method
        self.to_y = to_y
        self.scale = scale
        self.max_rgb = max_rgb

    def __call__(self, output, target):
        """Calculate sr metric.

        :param output: output of SR network
        :param target: ground truth from dataset
        :return: sr metric value
        """
        shape_list = output.get_shape().as_list()
        if len(shape_list) == 5:
            result = 0.
            for index in range(shape_list[4]):
                result += self.compute_metric(
                    output[:, :, :, :, index], target[:, :, :, :, index])
            sr_metric = result / shape_list[4]
        else:
            sr_metric = self.compute_metric(output, target)
        sr_metric = {self.__metric_name__: tf.compat.v1.metrics.mean(sr_metric)}
        return sr_metric

    def _preprocess(self, image):
        """Preprocess data."""
        if image.get_shape()[-1] != 1 and image.get_shape()[-1] != 3:
            image = tf.transpose(image, [0, 2, 3, 1])
        img_height, img_width = image.get_shape()[1:3]

        crop_height, crop_width = img_height - 2 * \
            self.scale, img_width - 2 * self.scale
        if self.max_rgb == 1:
            image = image * 255.0
        image = tf.image.crop_to_bounding_box(
            image, self.scale, self.scale, crop_height, crop_width)
        image = image / 255.0
        if self.to_y:
            multiplier = tf.constant([25.064 / 256.0, 129.057 / 256.0, 65.738 / 256.0],
                                     shape=[1, 1, 1, 3], dtype=image.dtype)
            image = tf.math.reduce_sum(image * multiplier, axis=3)
        return image

    def compute_metric(self, output, target):
        """Compute sr metric."""
        output, target = self._preprocess(output), self._preprocess(target)
        if self.method == 'ssim':
            sr_metric = tf.image.ssim(output, target, max_val=1.0)
        elif self.method == 'psnr':
            sr_metric = tf.image.psnr(output, target, max_val=1.0)
        return sr_metric
