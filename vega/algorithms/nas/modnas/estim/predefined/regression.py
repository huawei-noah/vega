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

"""Regression Estimator."""
import itertools
from modnas.core.param_space import ParamSpace
from modnas.registry.estim import register
from ..base import EstimBase


@register
class RegressionEstim(EstimBase):
    """Regression Estimator class."""

    def __init__(self, *args, predictor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = predictor

    def step(self, params):
        """Return evaluation results from remote Estimator."""
        ParamSpace().update_params(params)
        arch_desc = self.get_arch_desc()
        return self.predictor.predict(arch_desc)

    def run_epoch(self, optim, epoch, tot_epochs):
        """Run Estimator routine for one epoch."""
        config = self.config
        logger = self.logger
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        arch_batch_size = config.arch_update_batch
        # arch step
        if epoch >= tot_epochs:
            return {'stop': True}
        if not optim.has_next():
            logger.info('Search: finished')
            return {'stop': True}
        if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
            optim.step(self)
        inputs = optim.next(batch_size=arch_batch_size)
        self.clear_buffer()
        for params in inputs:
            # estim step
            self.stepped(params)
        self.wait_done()

    def run(self, optim):
        """Run Estimator routine."""
        config = self.config
        tot_epochs = config.epochs
        for epoch in itertools.count(self.cur_epoch + 1):
            if (self.run_epoch(optim, epoch=epoch, tot_epochs=tot_epochs) or {}).get('stop'):
                break
