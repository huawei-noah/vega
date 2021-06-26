# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of Regression task."""
from torch.nn import functional as F
from vega.metrics.pytorch.metrics import MetricBase
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.METRIC, alias='mse')
class MSE(MetricBase):
    """Calculate Mse accuracy between output and target."""

    __metric_name__ = 'mse'

    def __init__(self):
        """Init Mes metric."""
        self.sum = 0
        self.pfm = 0
        self.data_num = 0

    def __call__(self, output, target, *args, **kwargs):
        """Perform top k accuracy.

        :param output: output of classification network
        :param target: ground truth from dataset
        :return: pfm
        """
        res = 0 - F.mse_loss(output, target, reduction='mean').cpu().item()
        n = output.size(0)
        self.data_num += n
        self.sum += res * n
        self.pfm = self.sum / self.data_num
        return res

    @property
    def objective(self):
        """Define reward mode, default is max."""
        return 'MIN'

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.sum = 0
        self.data_num = 0
        self.pfm = None

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        return self.pfm
