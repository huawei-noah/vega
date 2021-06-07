# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metrics statistics reporter."""
import pickle
import itertools
from modnas.registry.callback import register
from modnas.callback.base import CallbackBase
from matplotlib import pyplot as plt
plt.switch_backend('Agg')


@register
class MetricsStatsReporter(CallbackBase):
    """Metrics statistics reporter class."""

    def __init__(self, axis_list=None):
        super().__init__({
            'after:EstimBase.step_done': self.on_step_done,
            'after:EstimBase.run': self.save_stats,
        })
        self.results = []
        self.axis_list = axis_list

    def on_step_done(self, ret, estim, params, value, arch_desc=None):
        """Record Estimator evaluation result on each step."""
        self.results.append((params, value))

    def save_stats(self, ret, estim, optim):
        """Save statistics on search end."""
        results = self.results
        if not results:
            return
        axis_list = self.axis_list
        if axis_list is None:
            metrics = list(results[0][1].keys())
            axis_list = list(itertools.combinations(metrics, r=2))
        self.logger.info('metrics stats: {} axis: {}'.format(len(results), axis_list))
        for i, axis in enumerate(axis_list):
            plt.figure(i)
            axis_str = '-'.join(axis)
            plt.title('metrics: {}'.format(axis_str))
            values = [[res[1][ax] for res in results] for ax in axis]
            plt.scatter(values[0], values[1])
            plt.xlabel(axis[0])
            plt.ylabel(axis[1])
            plt.savefig(estim.expman.join('plot', 'metrics_{}.png'.format(axis_str)))
        result_path = estim.expman.join('output', 'metrics_results.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)
            self.logger.info('metrics results saved to {}'.format(result_path))
        self.results = []
