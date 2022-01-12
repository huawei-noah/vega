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
