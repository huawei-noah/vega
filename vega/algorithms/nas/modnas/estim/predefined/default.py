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

"""Estimator with default training & evaluating methods."""
import itertools
from modnas.registry.estim import register
from ..base import EstimBase


@register
class DefaultEstim(EstimBase):
    """Default Estimator class."""

    def __init__(self, *args, save_best=True, valid_intv=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_best = save_best
        self.valid_intv = valid_intv

    def step(self, params):
        """Return evaluation results of a parameter set."""
        del params
        return self.compute_metrics()

    def run_epoch(self, optim, epoch, tot_epochs):
        """Run Estimator routine for one epoch."""
        if epoch == tot_epochs:
            return {'stop': True}
        # train
        self.train_epoch(epoch, tot_epochs)
        # valid
        if epoch + 1 == tot_epochs or (self.valid_intv is not None and not (epoch + 1) % self.valid_intv):
            self.stepped(None)
            self.wait_done()

    def run(self, optim):
        """Run Estimator routine."""
        self.reset_trainer()
        self.print_model_info()
        config = self.config
        tot_epochs = config.epochs
        for epoch in itertools.count(self.cur_epoch + 1):
            if (self.run_epoch(optim, epoch=epoch, tot_epochs=tot_epochs) or {}).get('stop'):
                break
