# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Estimator with default training & evaluating methods."""
import itertools
from ..base import EstimBase
from modnas.registry.estim import register


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
