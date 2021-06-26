# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of classifier task."""
from mindspore.nn.metrics import Accuracy
from vega.common import ClassFactory, ClassType
from vega.metrics.mindspore.metrics import MetricBase
import mindspore.nn as nn


@ClassFactory.register(ClassType.METRIC)
class accuracy(MetricBase):
    """Calculate classification accuracy between output and target."""

    __metric_name__ = 'accuracy'

    def __init__(self, topk=(1, 5)):
        """Init accuracy metric."""
        self.topk = topk

    def __call__(self, output, target, *args, **kwargs):
        """Forward and calculate accuracy."""
        if len(self.topk) == 1:
            return Accuracy()

        else:
            return {"accuracy": Accuracy(),
                    "accuracy_top1": nn.Top1CategoricalAccuracy(),
                    "accuracy_top5": nn.Top5CategoricalAccuracy()
                    }
