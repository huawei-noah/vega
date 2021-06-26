# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.


"""Metric of classifier task."""
from vega.metrics.pytorch.metrics import MetricBase
from vega.common import ClassFactory, ClassType
from sklearn.metrics import roc_auc_score


@ClassFactory.register(ClassType.METRIC, alias='auc')
class AUC(MetricBase):
    """Calculate roc_auc_score between output and target."""

    # _metric_name__ = 'auc'

    def __init__(self, **kwargs):
        """Init AUC metric."""
        self.pfm = 0.
        self.__metric_name__ = "auc"
        print("init roc_auc_score metric finish")

    def __call__(self, output, target, *args, **kwargs):
        """Call auc metric calculate."""
        output = output.tolist()
        target = target.tolist()
        # print("output:", len(output))
        res = roc_auc_score(y_score=output, y_true=target)
        self.pfm = res
        # print("auc metrics:", res)
        return res

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.pfm = 0.

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        return self.pfm
