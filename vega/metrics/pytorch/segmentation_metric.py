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
import numpy as np
from vega.metrics.pytorch.metrics import MetricBase
from vega.common import ClassFactory, ClassType


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
    logits = output.detach().cpu().numpy().transpose((0, 2, 3, 1))
    preds = logits.argmax(axis=3).reshape(-1)
    mask = mask.detach().cpu().numpy().reshape(-1)
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
class IoUMetric(MetricBase):
    """Calculate IoU between output and target."""

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_sum = np.zeros((num_class, num_class))

    def __call__(self, output, target, *args, **kwargs):
        """Calculate confusion matrix.

        :param output: output of segmentation network
        :param target: ground truth from dataset
        :return: confusion matrix sum
        """
        if isinstance(output, list):
            confusion = calc_confusion_matrix(output[-1], target, self.num_class)
        else:
            confusion = calc_confusion_matrix(output, target, self.num_class)
        self.confusion_sum += confusion
        return confusion

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.confusion_sum = np.zeros((self.num_class, self.num_class))

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        iou = compute_iou(self.confusion_sum).mean()
        return iou
