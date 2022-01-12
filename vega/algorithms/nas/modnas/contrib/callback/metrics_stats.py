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

"""Metrics statistics reporter."""
import itertools
from collections import OrderedDict
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from modnas.registry.callback import register
from modnas.callback.base import CallbackBase
from modnas.estim.base import EstimBase
from modnas.optim.base import OptimBase
from vega.common import FileOps

plt.switch_backend('Agg')


@register
class MetricsStatsReporter(CallbackBase):
    """Metrics statistics reporter class."""

    def __init__(self, axis_list: List[Tuple[int, int]] = None) -> None:
        super().__init__({
            'after:EstimBase.step_done': self.on_step_done,
            'after:EstimBase.run': self.save_stats,
        })
        self.results = []
        self.axis_list = axis_list

    def on_step_done(
        self, ret: Dict[str, bool], estim: EstimBase, params: Optional[OrderedDict],
        value: Dict[str, float], arch_desc: Optional[Any] = None
    ) -> None:
        """Record Estimator evaluation result on each step."""
        self.results.append((params, value))

    def save_stats(self, ret: Dict[str, Any], estim: EstimBase, optim: OptimBase) -> Dict[str, Any]:
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
        FileOps.dump_pickle(results, result_path)
        self.logger.info('metrics results saved to {}'.format(result_path))
        self.results = []
        return ret
