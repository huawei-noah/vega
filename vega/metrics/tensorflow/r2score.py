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
