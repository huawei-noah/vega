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
