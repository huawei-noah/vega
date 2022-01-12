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

"""Unified Estimator."""
import itertools
from collections import OrderedDict
from typing import Dict, Optional, Any
from modnas.core.param_space import ParamSpace
from modnas.registry.estim import register
from modnas.optim.base import OptimBase
from ..base import EstimBase


@register
class UnifiedEstim(EstimBase):
    """Unified Estimator class."""

    def __init__(
        self, train_epochs: int = 1, train_steps: int = -1, reset_training: bool = False, eval_steps: int = 1,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if train_steps != 0:
            train_epochs = 1
        self.train_epochs = train_epochs
        self.train_steps = train_steps
        self.reset_training = reset_training
        self.eval_steps = eval_steps
        self.cur_step = -1

    def step(self, params: OrderedDict) -> Dict[str, Any]:
        """Return evaluation results of a parameter set."""
        ParamSpace().update_params(params)
        n_train_batch = self.get_num_train_batch()
        n_valid_batch = self.get_num_valid_batch()
        train_epochs = self.train_epochs
        train_steps = self.train_steps
        if train_steps == 0:
            train_steps = n_train_batch
        elif train_steps == -1:
            train_steps = max(round(n_train_batch / (n_valid_batch or 1)), 1)
        if self.reset_training:
            self.reset_trainer(epochs=train_epochs)
        for epoch in range(train_epochs):
            for _ in range(train_steps):
                self.cur_step += 1
                if self.cur_step >= n_train_batch:
                    self.cur_step = -1
                    break
                self.train_step(model=self.model,
                                epoch=epoch,
                                tot_epochs=train_epochs,
                                step=self.cur_step,
                                tot_steps=n_train_batch)
        if (self.cur_step + 1) % self.eval_steps != 0:
            return {'default': None}
        arch_desc = self.exporter(self.model)
        ret = self.compute_metrics()
        self.logger.info('Evaluate: {} -> {}'.format(arch_desc, ret))
        return ret

    def run_epoch(self, optim: OptimBase, epoch: int, tot_epochs: int) -> Optional[Dict[str, Any]]:
        """Run Estimator routine for one epoch."""
        logger = self.logger
        config = self.config
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        arch_batch_size = config.arch_update_batch
        train_steps = self.train_steps
        n_epoch_steps = 1 if train_steps == 0 else (self.get_num_train_batch() + train_steps - 1) // train_steps
        if self.cur_epoch >= tot_epochs:
            return {'stop': True}
        # arch step
        if not optim.has_next():
            logger.info('Search: finished')
            return {'stop': True}
        if self.cur_epoch >= arch_epoch_start and (self.cur_epoch - arch_epoch_start) % arch_epoch_intv == 0:
            optim.step(self)
        self.inputs = optim.next(batch_size=arch_batch_size)
        self.clear_buffer()
        self.batch_best = None
        for params in self.inputs:
            # estim step
            self.stepped(params)
        self.wait_done()
        if (epoch + 1) % n_epoch_steps != 0:
            return None
        self.cur_epoch += 1
        return None

    def run(self, optim: OptimBase) -> None:
        """Run Estimator routine."""
        self.reset_trainer()
        config = self.config
        tot_epochs = config.epochs
        self.cur_epoch += 1
        for epoch in itertools.count(0):
            if (self.run_epoch(optim, epoch=epoch, tot_epochs=tot_epochs) or {}).get('stop'):
                break
