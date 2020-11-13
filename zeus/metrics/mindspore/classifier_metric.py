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
from zeus.common import ClassFactory, ClassType
from zeus.metrics.mindspore.metrics import MetricBase


@ClassFactory.register(ClassType.METRIC)
class accuracy(MetricBase):
    """Calculate classification accuracy between output and target."""

    __metric_name__ = 'accuracy'

    def __init__(self, topk=(1,)):
        """Init accuracy metric."""
        pass

    def __call__(self, output, target, *args, **kwargs):
        """Forward and calculate accuracy."""
        return Accuracy()
