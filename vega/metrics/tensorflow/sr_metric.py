# -*- coding:utf-8 -*-

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

"""Metric of super resolution task."""
import tensorflow as tf
from vega.common import ClassFactory, ClassType
from vega.metrics.tensorflow.metrics import MetricBase


def preprocess(image, scale=2, max_rgb=1, to_y=True):
    """Preprocess data."""
    if image.get_shape()[-1] != 1 and image.get_shape()[-1] != 3:
        image = tf.transpose(image, [0, 2, 3, 1])
    img_height, img_width = image.get_shape()[1:3]

    crop_height, crop_width = img_height - 2 * scale, img_width - 2 * scale
    if max_rgb == 1:
        image = image * 255.0
    image = tf.image.crop_to_bounding_box(
        image, scale, scale, crop_height, crop_width)
    image = image / 255.0
    if to_y:
        multiplier = tf.constant([25.064 / 256.0, 129.057 / 256.0, 65.738 / 256.0],
                                 shape=[1, 1, 1, 3], dtype=image.dtype)
        image = tf.math.reduce_sum(image * multiplier, axis=3)
    return image


@ClassFactory.register(ClassType.METRIC)
class PSNR(MetricBase):
    """Calculate IoU between output and target."""

    __metric_name__ = 'PSNR'

    def __init__(self, to_y=True, scale=2, max_rgb=1):
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
            output = preprocess(output, self.scale, self.max_rgb, self.to_y)
            target = preprocess(target, self.scale, self.max_rgb, self.to_y)

            sr_metric = tf.image.psnr(output, target, max_val=1.0)
        sr_metric = {self.__metric_name__: tf.compat.v1.metrics.mean(sr_metric)}
        return sr_metric


@ClassFactory.register(ClassType.METRIC)
class SSIM(MetricBase):
    """Calculate IoU between output and target."""

    __metric_name__ = 'SSIM'

    def __init__(self, to_y=True, scale=2, max_rgb=1):
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
            output = preprocess(output, self.scale, self.max_rgb, self.to_y)
            target = preprocess(target, self.scale, self.max_rgb, self.to_y)

            sr_metric = tf.image.ssim(output, target, max_val=1.0)
        sr_metric = {self.__metric_name__: tf.compat.v1.metrics.mean(sr_metric)}
        return sr_metric
