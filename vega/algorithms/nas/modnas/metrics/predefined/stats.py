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

"""Statistical metrics."""
from typing import List, Any, Optional
import yaml
import numpy as np
from modnas.registry.metrics import register, build
from modnas.registry import SPEC_TYPE
from vega.common import FileOps
from ..base import MetricsBase


@register
class StatsLUTMetrics(MetricsBase):
    """Statistical metrics using look-up table (LUT)."""

    def __init__(self, lut_path: str, head: List[str]) -> None:
        super().__init__()
        with open(lut_path, 'r') as f:
            self.lut = yaml.safe_load(f)
        if self.lut is None:
            raise ValueError('StatsLUT: Error loading LUT: {}'.format(lut_path))
        self.head = head
        self.warned = set()

    def __call__(self, stats: Any) -> float:
        """Return metrics output."""
        key = '#'.join([str(stats[k]) for k in self.head if not stats.get(k, None) is None])
        val = self.lut.get(key, None)
        if val is None:
            if key not in self.warned:
                self.logger.warning('StatsLUT: missing key in LUT: {}'.format(key))
                self.warned.add(key)
        elif isinstance(val, dict):
            val = float(np.random.normal(val['mean'], val['std']))
        else:
            val = float(val)
        return val


@register
class StatsRecordMetrics(MetricsBase):
    """Statistical metrics using recorded results."""

    def __init__(self, metrics: SPEC_TYPE, head: List[str], save_path: Optional[str] = None) -> None:
        super().__init__()
        self.head = head
        self.metrics = build(metrics)
        self.record = dict()
        self.save_path = save_path
        self.save_file = None
        if save_path is not None:
            self.save_file = open(save_path, 'w')

    def __call__(self, stats: Any) -> float:
        """Return metrics output."""
        key = '#'.join([str(stats[k]) for k in self.head if stats[k] is not None])
        if key in self.record:
            return self.record[key]
        val = self.metrics(stats)
        self.record[key] = val
        self.logger.info('StatsRecord:\t{}: {}'.format(key, val))
        if self.save_file is not None:
            self.save_file.write('{}: {}\n'.format(key, val))
        return val


@register
class StatsModelMetrics(MetricsBase):
    """Statistical metrics using predictor."""

    def __init__(self, model_path, head):
        super().__init__()
        self.model = FileOps.load_pickle(model_path)
        self.head = head

    def __call__(self, stats):
        """Return metrics output."""
        feats = [stats.get(c, None) for c in self.head]
        return self.model.predict(feats)
