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
        self.best_score = None
        self.valid_intv = valid_intv

    def step(self, params):
        """Return evaluation results of a parameter set."""
        del params
        return self.compute_metrics()

    def run_epoch(self, optim, epoch, tot_epochs):
        """Run Estimator routine for one epoch."""
        config = self.config
        if epoch == tot_epochs:
            return 1
        # train
        self.train_epoch(epoch, tot_epochs)
        # valid
        if epoch + 1 == tot_epochs or (self.valid_intv is not None and not (epoch + 1) % self.valid_intv):
            val_score = self.get_score(self.step(None))
        else:
            val_score = None
        # save
        if val_score is not None and (self.best_score is None or val_score > self.best_score):
            self.best_score = val_score
            if self.save_best:
                self.save_checkpoint(epoch, save_name='best')
        if config.save_freq != 0 and epoch % config.save_freq == 0:
            self.save_checkpoint(epoch)
        return {
            'epoch_best': val_score,
        }

    def run(self, optim):
        """Run Estimator routine."""
        self.reset_trainer()
        self.print_model_info()
        config = self.config
        tot_epochs = config.epochs
        for epoch in itertools.count(self.cur_epoch + 1):
            if self.run_epoch(optim, epoch=epoch, tot_epochs=tot_epochs) == 1:
                break
        return {'best_score': self.best_score}
