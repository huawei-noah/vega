# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of Regression task."""
import numpy as np
from scipy import stats
import torch
from torch.nn import functional as F
from vega.metrics.pytorch.metrics import MetricBase
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.METRIC, alias='r2score')
class R2Score(MetricBase):
    """Calculate R2 Score between output and target."""

    __metric_name__ = 'r2score'

    def __init__(self):
        """Init R2 Score metric."""
        self.ess = 0
        self.tss = 0
        self.sum = 0
        self.num = 0
        self.pfm = 0
        self.targets = None
        self.outputs = None

    def __call__(self, output, target, *args, **kwargs):
        """Forward and calculate r2 score."""
        output, target = torch.squeeze(output), torch.squeeze(target)
        temp_mean = torch.sum(target) / target.size(0)
        temp_ess = F.mse_loss(target, output, reduction='sum').cpu().item()
        temp_tss = F.mse_loss(target, temp_mean.repeat(target.size(0)), reduction='sum').cpu().item()
        temp_r2_score = 1 - temp_ess / temp_tss
        self.sum += torch.sum(target)
        self.num += target.size(0)
        mean = self.sum / self.num
        self.ess += temp_ess
        if self.targets is not None:
            self.targets = torch.cat((self.targets, target), dim=0)
        else:
            self.targets = target
        self.tss = F.mse_loss(self.targets, mean.repeat(self.num), reduction='sum').cpu().item()
        self.pfm = 1 - self.ess / self.tss
        return temp_r2_score

    @property
    def objective(self):
        """Define reward mode, default is max."""
        return 'MAX'

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.ess = 0
        self.tss = 0
        self.sum = 0
        self.num = 0
        self.pfm = 0
        self.targets = None
        self.outputs = None

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        return self.pfm
