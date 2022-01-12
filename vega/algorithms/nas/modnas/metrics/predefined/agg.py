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

"""Aggregate metrics."""
from functools import reduce
from typing import Dict, Any
from modnas.registry.metrics import register, build
from ..base import MetricsBase


@register
class SumAggMetrics(MetricsBase):
    """Aggregate metrics by sum."""

    def __init__(self, metrics_conf: Dict) -> None:
        super().__init__()
        self.metrics = {k: build(conf) for k, conf in metrics_conf.items()}
        self.base = {k: conf.get('base', 1) for k, conf in metrics_conf.items()}
        self.weight = {k: conf.get('weight', 1) for k, conf in metrics_conf.items()}

    def __call__(self, item: Any) -> Any:
        """Return metrics output."""
        mt_res = {k: (mt(item) or 0) for k, mt in self.metrics.items()}
        self.logger.info('SumAgg: {{{}}}'.format(', '.join(['{}: {}'.format(k, r) for k, r in mt_res.items()])))
        return sum(self.weight[k] * mt_res[k] / self.base[k] for k in self.metrics)


@register
class ProdAggMetrics(MetricsBase):
    """Aggregate metrics by product."""

    def __init__(self, metrics_conf):
        super().__init__()
        self.metrics = {k: build(conf) for k, conf in metrics_conf.items()}
        self.base = {k: conf.get('base', 1) for k, conf in metrics_conf.items()}
        self.alpha = {k: conf.get('alpha', 1) for k, conf in metrics_conf.items()}
        self.beta = {k: conf.get('beta', 1) for k, conf in metrics_conf.items()}

    def __call__(self, item):
        """Return metrics output."""
        mt_res = {k: (mt(item) or 0) for k, mt in self.metrics.items()}
        self.logger.info('ProdAgg: {{{}}}'.format(', '.join(['{}: {}'.format(k, r) for k, r in mt_res.items()])))
        mt_w = [(mt_res[k] / self.base[k])**(self.beta[k] if mt_res[k] > self.base[k] else self.alpha[k])
                for k in self.metrics]
        return reduce(lambda x, y: x * y, mt_w)


@register
class MergeAggMetrics(MetricsBase):
    """Aggregate metrics by merging dict."""

    def __init__(self, metrics_conf):
        super().__init__()
        self.metrics = {k: build(conf) for k, conf in metrics_conf.items()}

    def __call__(self, item):
        """Return metrics output."""
        return {k: mt(item) for k, mt in self.metrics.items()}
