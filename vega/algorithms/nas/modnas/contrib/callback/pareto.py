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

"""Pareto optimum statistics reporter."""
from modnas.registry.callback import register
from modnas.registry.callback import OptimumReporter
from matplotlib import pyplot as plt
plt.switch_backend('Agg')


@register
class ParetoReporter(OptimumReporter):
    """Pareto optimum statistics reporter class."""

    def __init__(self, *args, plot_keys=None, plot_args=None, plot_intv=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_keys = plot_keys
        self.plot_args = plot_args
        self.plot_intv = plot_intv

    def plot_pareto(self, estim, epoch=None):
        """Plot pareto optimum."""
        if not self.results or not self.opt_results:
            return
        plt.figure()
        plt.title('Pareto optimum')
        plot_keys = self.plot_keys or self.cmp_keys or list(self.results[0][1].keys())
        plot_keys = plot_keys[:2][::-1]
        if len(plot_keys) < 2:
            self.logger.error('Not enough metrics for pareto plot')
            return
        domed_res = [r for r in self.results if r not in self.opt_results]
        vals = [[m.get(k, 0) for _, m in domed_res] for k in plot_keys]
        opt_vals = [[m.get(k, 0) for _, m in self.opt_results] for k in plot_keys]
        plt.scatter(*vals, **(self.plot_args or {}))
        plt.scatter(*opt_vals, **(self.plot_args or {}))
        plt.xlabel(plot_keys[0])
        plt.ylabel(plot_keys[1])
        plot_path = estim.expman.join('plot', 'pareto{}.png'.format('' if epoch is None else ('_' + str(epoch))))
        plt.savefig(plot_path)
        self.logger.info('Pareto plot saved to {}'.format(plot_path))

    def report_epoch(self, ret, estim, optim, epoch, tot_epochs):
        """Plot pareto optimum on epochs."""
        intv = self.plot_intv
        if intv and intv < 1:
            intv = int(intv * tot_epochs)
        if intv is not None and (epoch + 1) % intv == 0:
            self.plot_pareto(estim, epoch + 1)
        return super().report_epoch(ret, estim, optim, epoch, tot_epochs)

    def report_results(self, ret, estim, optim):
        """Plot pareto optimum on search end."""
        self.plot_pareto(estim)
        return super().report_results(ret, estim, optim)
