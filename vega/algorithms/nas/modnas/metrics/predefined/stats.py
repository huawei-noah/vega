# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Statistical metrics."""
import yaml
import pickle
import numpy as np
from ..base import MetricsBase
from modnas.registry.metrics import register, build


@register
class StatsLUTMetrics(MetricsBase):
    """Statistical metrics using look-up table (LUT)."""

    def __init__(self, lut_path, head=None):
        super().__init__()
        with open(lut_path, 'r') as f:
            self.lut = yaml.load(f, Loader=yaml.Loader)
        if self.lut is None:
            raise ValueError('StatsLUT: Error loading LUT: {}'.format(lut_path))
        self.head = head
        self.warned = set()

    def __call__(self, stats):
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

    def __init__(self, metrics, head=None, save_path=None):
        super().__init__()
        self.head = head
        self.metrics = build(metrics)
        self.record = dict()
        self.save_path = save_path
        self.save_file = None
        if save_path is not None:
            self.save_file = open(save_path, 'w')

    def __call__(self, stats):
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
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.head = head

    def __call__(self, stats):
        """Return metrics output."""
        feats = [stats.get(c, None) for c in self.head]
        return self.model.predict(feats)
