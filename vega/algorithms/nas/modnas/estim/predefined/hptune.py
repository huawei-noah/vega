# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Hyperparameter-tuning Estimator."""

import copy
import itertools
import traceback
from multiprocessing import Process, Pipe
from ..base import EstimBase
from modnas.utils.config import Config
from modnas.registry.runner import build as build_runner
from modnas.registry.estim import register


def _default_trial_runner(conn, trial_proc, trial_args):
    ret = build_runner(trial_proc, **trial_args)
    conn.send(ret)


@register
class HPTuneEstim(EstimBase):
    """Hyperparameter-tuning Estimator class."""

    def __init__(self,
                 measure_fn=None,
                 batch_size=1,
                 early_stopping=None,
                 trial_proc=None,
                 trial_config=None,
                 trial_args=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.measure_fn = measure_fn or self._default_measure_fn
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.trial_proc = trial_proc
        self.trial_config = trial_config
        self.trial_args = trial_args
        self.best_hparams = None
        self.best_score = None
        self.best_iter = 0
        self.trial_index = 0

    def _default_measure_fn(self, hp, **kwargs):
        trial_config = copy.deepcopy(Config.load(self.trial_config))
        Config.apply(trial_config, hp)
        trial_args = dict(copy.deepcopy(self.trial_args))
        trial_args['name'] = '{}_{}'.format(trial_args.get('name', 'trial'), self.trial_index)
        trial_args['config'] = trial_config.to_dict()
        p_con, c_con = Pipe()
        proc = Process(target=_default_trial_runner, args=(c_con, self.trial_proc, trial_args))
        proc.start()
        proc.join()
        if not p_con.poll(0):
            return 0
        ret = p_con.recv()
        return ret['final'].get('best_score', list(ret.values())[0])

    def step(self, hp):
        """Return evaluation results of a parameter set."""
        self.trial_index += 1
        logger = self.logger
        logger.info('measuring hparam: {}'.format(hp))
        config = self.config
        fn_args = config.get('trial_args', {})
        try:
            score = self.measure_fn(hp, **fn_args)
            error_no = 0
        except RuntimeError:
            score = 0
            error_no = -1
            logger.info('trial {} failed with error: {}'.format(self.trial_index, traceback.format_exc()))
        result = {
            'score': score or 0,
            'error_no': error_no,
        }
        return result

    def run_epoch(self, optim, epoch, tot_epochs):
        """Run Estimator routine for one epoch."""
        batch_size = self.batch_size
        early_stopping = self.early_stopping
        if epoch >= tot_epochs:
            return 1
        if not optim.has_next():
            self.logger.info('HPTune: all finished')
            return 1
        inputs = optim.next(batch_size)
        self.clear_buffer()
        for hp in inputs:
            res = self.stepped(hp)
        self.wait_done()
        for hp, res, _ in self.buffer():
            score = 0 if res['error_no'] else res['score']
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_hparams = hp
                self.best_iter = epoch
        optim.step(self)
        if early_stopping is not None and epoch >= self.best_iter + early_stopping:
            self.logger.info('HPTune: early stopped: {}'.format(epoch))
            return 1

    def run(self, optim):
        """Run Estimator routine."""
        config = self.config
        tot_epochs = config.epochs
        for epoch in itertools.count(self.cur_epoch + 1):
            if self.run_epoch(optim, epoch, tot_epochs) == 1:
                break
        return {
            'best_iter': self.best_iter,
            'best_score': self.best_score,
            'best_hparams': self.best_hparams,
        }
