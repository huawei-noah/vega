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

"""Search optimum statistics reporter."""
from functools import partial
from modnas.utils import format_value
from modnas.registry.callback import register
from modnas.callback.base import CallbackBase


def MIN_CMP(x, y):
    """Return min comparison result."""
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return 0
    return y - x


def MAX_CMP(x, y):
    """Return max comparison result."""
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return 0
    return x - y


@register
class OptimumReporter(CallbackBase):
    """Search optimum statistics reporter class."""

    priority = 0

    def __init__(self, cmp_keys=None, cmp_fn=None, cmp_th=None, score_fn=None, format_fn=None, stat_epoch=True):
        handlers = {
            'before:EstimBase.run': self.reset,
            'after:EstimBase.step_done': self.on_step_done,
            'after:EstimBase.run': self.report_results,
        }
        if stat_epoch:
            handlers['after:EstimBase.run_epoch'] = self.report_epoch
        super().__init__(handlers)
        self.stat_epoch = stat_epoch
        self.cmp_keys = cmp_keys
        cmp_fn = cmp_fn or {}
        cmp_fn = {k: (MAX_CMP if v == 'max' else MIN_CMP if v == 'min' else v) for k, v in cmp_fn.items()}
        self.cmp_fn = cmp_fn or {}
        self.cmp_th = cmp_th or {}
        self.score_fn = score_fn
        self.format_fn = format_fn or partial(format_value, unit=False, factor=0, prec=4)
        self.results = []
        self.opt_results = []
        self.ep_opt_results = []

    def reset(self, estim, optim):
        """Reset callback states."""
        self.results = []
        self.opt_results = []
        self.ep_opt_results = []
        self.cur_cmp_keys = self.cmp_keys

    def update_optimal(self, res, opts):
        """Update current optimal results."""
        met = res[1]
        if self.cur_cmp_keys is None:
            self.cur_cmp_keys = list(met.keys())
        rem_opt = []
        for i, (_, m) in enumerate(opts):
            c = self.dom_cmp(met, m)
            if c < 0:
                return opts
            elif c > 0:
                rem_opt.append(i)
        opts = [r for i, r in enumerate(opts) if i not in rem_opt]
        opts.append(res)
        return opts

    def dom_cmp(self, m1, m2):
        """Return dominating comparison between metrics."""
        dom = 0
        for k in self.cur_cmp_keys:
            v1, v2 = m1.get(k, None), m2.get(k, None)
            cmp = self.cmp_fn.get(k, MAX_CMP)(v1, v2)
            th = self.cmp_th.get(k, 0)
            if cmp > th:
                if dom == -1:
                    return 0
                dom = 1
            elif cmp < -th:
                if dom == 1:
                    return 0
                dom = -1
        return dom

    def on_step_done(self, ret, estim, params, value, arch_desc=None):
        """Record Estimator evaluation result on each step."""
        ret = ret or {}
        if params is False or ret.get('no_opt'):
            return
        if self.score_fn:
            value = {'score': self.score_fn(value)}
        if not isinstance(value, dict):
            value = {'default': value}
        arch_desc = arch_desc or estim.get_arch_desc()
        res = ((None if params is None or not isinstance(params, dict) else (arch_desc or dict(params))), value)
        self.results.append(res)
        self.opt_results = self.update_optimal(res, self.opt_results)
        if self.stat_epoch:
            self.ep_opt_results = self.update_optimal(res, self.ep_opt_results)
        if res in self.opt_results:
            ret['is_opt'] = True
        return ret

    def format_metrics(self, opts):
        """Format metrics."""
        if not opts:
            return None
        met = [r[1] for r in opts]
        met = [{k: self.format_fn(v) for k, v in m.items()} for m in met]
        met = [(list(m.values())[0] if len(m) == 1 else m) for m in met]
        if len(met) == 1:
            met = met[0]
        return met

    def report_epoch(self, ret, estim, optim, epoch, tot_epochs):
        """Report optimum in each epoch."""
        ret = ret or {}
        if self.ep_opt_results:
            ret['epoch_opt'] = self.format_metrics(self.ep_opt_results)
        if self.opt_results:
            ret['opt'] = self.format_metrics(self.opt_results)
        self.ep_opt_results = []
        return ret

    def report_results(self, ret, estim, optim):
        """Report optimum on search end."""
        opt_res = {}
        if self.opt_results:
            opt_res['opt_results'] = self.opt_results
        ret = ret or {}
        ret.update(opt_res)
        return ret
