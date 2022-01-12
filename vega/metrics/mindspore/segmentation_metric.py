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

"""Metric of segmentation task."""
from mindspore.nn.metrics import Metric
from vega.common import ClassFactory, ClassType
import numpy as np


def calc_confusion_matrix(output, mask, num_class):
    """Calculate confusion matrix between output and mask.

    :param output: predicted images
    :type output: pytorch tensor
    :param mask: images of ground truth
    :type mask: pytorch tensor
    :return: confusion matrix
    :rtype: numpy matrix
    """
    confusion_matrix = np.zeros((num_class, num_class))
    preds = output.argmax(axis=3).reshape(-1)
    mask = mask.reshape(-1)
    for predicted, label in zip(preds, mask):
        if label < num_class:
            confusion_matrix[predicted][label] += 1
    return confusion_matrix


def compute_iou(confusion_matrix):
    """Compute IU from confusion matrix.

    :param confusion_matrix: square confusion matrix.
    :type confusion_matrix: numpy matrix
    :return: IU vector.
    :rtype: numpy vector
    """
    n_classes = confusion_matrix.shape[0]
    IoU = np.zeros(n_classes)
    for i in range(n_classes):
        sum_columns = np.sum(confusion_matrix[:, i])
        sum_rows = np.sum(confusion_matrix[i, :])
        num_correct = confusion_matrix[i, i]
        denom = sum_columns + sum_rows - num_correct
        if denom > 0:
            IoU[i] = num_correct / denom
    return IoU


@ClassFactory.register(ClassType.METRIC, alias='IoUMetric')
class IoUMetric(Metric):
    """Calculate IoU between output and target."""

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_sum = np.zeros((num_class, num_class))

    def update(self, *inputs):
        """Update the metric."""
        if len(inputs) != 2:
            raise ValueError('IoUMetric need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        self.result = self.compute_metric(y_pred, y)

    def eval(self):
        """Get the metric."""
        return self.result

    def clear(self):
        """Reset the metric."""
        self.result = 0

    def compute_metric(self, output, target):
        """Compute sr metric."""
        confusion = calc_confusion_matrix(output, target, self.num_class)
        self.confusion_sum += confusion
        iou = compute_iou(self.confusion_sum).mean()
        return iou

    @property
    def objective(self):
        """Define reward mode, default is max."""
        return 'MAX'

    def __call__(self, output, target, *args, **kwargs):
        """Calculate confusion matrix.

        :param output: output of segmentation network
        :param target: ground truth from dataset
        :return: confusion matrix sum
        """
        return self
