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

"""Estimator results exporter."""
from modnas.registry.callback import register
from ..base import CallbackBase


@register
class EstimResultsExporter(CallbackBase):
    """Estimator results exporter class."""

    priority = -1

    def __init__(self, run_file_name='results', best_file_name='best',
                 chkpt_intv=0, desc_intv=0, save_chkpt_best=True, save_desc_best=True):
        super().__init__({
            'after:EstimBase.step_done': self.on_step_done,
            'after:EstimBase.run': self.export_run,
            'after:EstimBase.run_epoch': self.export_epoch,
        })
        self.run_file_name = run_file_name
        self.best_file_name = best_file_name
        self.chkpt_intv = chkpt_intv
        self.desc_intv = desc_intv
        self.save_chkpt_best = save_chkpt_best
        self.save_desc_best = save_desc_best

    def on_step_done(self, ret, estim, params, value, arch_desc=None):
        """Export result on each step."""
        if (ret or {}).get('is_opt'):
            if self.save_chkpt_best:
                estim.save_checkpoint(save_name=self.best_file_name)
            if params is not None:
                arch_desc = arch_desc or estim.get_arch_desc()
                if self.save_desc_best:
                    estim.save_arch_desc(save_name=self.best_file_name, arch_desc=arch_desc)

    def export_run(self, ret, estim, *args, **kwargs):
        """Export results after run."""
        best_res = {}
        ret = ret or {}
        opts = ret.get('opt_results')
        if opts:
            best_res['best_arch_desc'] = opts[-1][0]
            best_score = opts[-1][1]
            if isinstance(best_score, dict) and len(best_score) == 1:
                best_score = list(best_score.values())[0]
            best_res['best_score'] = best_score
            if len(opts) == 1:
                ret.pop('opt_results')
        ret.update(best_res)
        estim.save_arch_desc(save_name=self.run_file_name, arch_desc=ret)
        return ret

    def export_epoch(self, ret, estim, optim, epoch, tot_epochs):
        """Export results in each epoch."""
        if epoch >= tot_epochs:
            return
        if self.desc_intv and epoch % self.desc_intv == 0:
            estim.save_arch_desc(epoch)
        if self.chkpt_intv and epoch % self.chkpt_intv == 0:
            estim.save_checkpoint(epoch)
        return ret
