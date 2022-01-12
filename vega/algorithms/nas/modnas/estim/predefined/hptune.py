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

"""Hyperparameter-tuning Estimator."""
import copy
import itertools
import traceback
import multiprocessing as mp
import yaml
from modnas.utils.config import Config
from modnas.utils.wrapper import run
from modnas.registry.estim import register
from ..base import EstimBase


def _default_trial_runner(conn, trial_args):
    ret = run(**(yaml.load(trial_args, Loader=yaml.SafeLoader) or {}))
    conn.send(ret)


@register
class HPTuneEstim(EstimBase):
    """Hyperparameter-tuning Estimator class."""

    def __init__(self,
                 measure_fn=None,
                 batch_size=1,
                 early_stopping=None,
                 trial_config=None,
                 trial_args=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.measure_fn = measure_fn or self._default_measure_fn
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.trial_config = trial_config
        self.trial_args = trial_args
        self.best_hparams = None
        self.best_score = None
        self.best_iter = 0
        self.trial_index = 0
        self.is_succ = False

    def _default_measure_fn(self, hp, **kwargs):
        trial_config = copy.deepcopy(Config.load(self.trial_config))
        Config.apply(trial_config, hp)
        trial_args = dict(copy.deepcopy(self.trial_args))
        trial_args['name'] = '{}_{}'.format(trial_args.get('name', 'trial'), self.trial_index)
        trial_args['config'] = trial_config.to_dict()
        ctx = mp.get_context('spawn')
        p_con, c_con = ctx.Pipe()
        proc = ctx.Process(target=_default_trial_runner, args=(c_con, yaml.dump(trial_args)))
        proc.start()
        proc.join()
        if not p_con.poll(0):
            return 0
        ret = p_con.recv()
        ret = ret.get('final', list(ret.values())[-1])
        return ret.get('best_score', list(ret.values())[0])

    def step(self, hp):
        """Return evaluation results of a parameter set."""
        self.trial_index += 1
        logger = self.logger
        config = self.config
        fn_args = config.get('trial_args', {})
        try:
            score = self.measure_fn(hp, **fn_args)
            self.is_succ = True
        except RuntimeError as e:
            score = 0
            logger.debug(traceback.format_exc())
            logger.info(f'trial {self.trial_index} failed with error, message: {e}')
        result = {
            'score': score,
        }
        logger.info('Evaluate hparam: {} -> {}'.format(hp, result))
        return result

    def run_epoch(self, optim, epoch, tot_epochs):
        """Run Estimator routine for one epoch."""
        batch_size = self.batch_size
        early_stopping = self.early_stopping
        if tot_epochs != -1 and epoch >= tot_epochs:
            return {'stop': True}
        if not optim.has_next():
            self.logger.info('HPTune: all finished')
            return {'stop': True}
        inputs = optim.next(batch_size)
        self.clear_buffer()
        for hp in inputs:
            res = self.stepped(hp)
        self.wait_done()
        for hp, res, _ in self.buffer():
            score = res['score']
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_iter = epoch
        optim.step(self)
        if early_stopping is not None and epoch >= self.best_iter + early_stopping:
            self.logger.info('HPTune: early stopped: {}'.format(epoch))
            return {'stop': True}

    def run(self, optim):
        """Run Estimator routine."""
        config = self.config
        tot_epochs = config.epochs
        for epoch in itertools.count(self.cur_epoch + 1):
            if (self.run_epoch(optim, epoch, tot_epochs) or {}).get('stop'):
                break
        if not self.is_succ:
            raise RuntimeError('All trials failed')
