# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of segmentation task."""
from mindspore.nn.metrics import Metric
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.METRIC, alias='IoUMetric')
class IoUMetric(Metric):
    """Calculate IoU between output and target."""

    def __init__(self, num_class):
        self.num_class = num_class

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
        # TODO
        return 0

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
