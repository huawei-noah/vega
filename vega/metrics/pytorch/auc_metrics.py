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
from vega.metrics.pytorch.metrics import MetricBase
from vega.common import ClassFactory, ClassType
from sklearn.metrics import roc_auc_score


@ClassFactory.register(ClassType.METRIC, alias='auc')
class AUC(MetricBase):
    """Calculate roc_auc_score between output and target."""

    def __init__(self, **kwargs):
        """Init AUC metric."""
        self.pfm = 0.
        self.__metric_name__ = "auc"
        print("init roc_auc_score metric finish")

    def __call__(self, output, target, *args, **kwargs):
        """Call auc metric calculate."""
        output = output.tolist()
        target = target.tolist()
        res = roc_auc_score(y_score=output, y_true=target)
        self.pfm = res
        return res

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.pfm = 0.

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        return self.pfm
