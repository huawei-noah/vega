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

"""ModularNAS framework trainer callback."""

import copy
import logging
import threading
import traceback
from vega.common import FileOps
from vega.common import ClassFactory, ClassType
from vega.trainer.callbacks import Callback
from vega.report import ReportClient
from vega.core.search_space import SearchSpace
from vega.core.search_algs import SearchAlgorithm
from modnas.data_provider.predefined.default import DefaultDataProvider
from modnas.trainer.base import TrainerBase
from modnas.utils.wrapper import init_all
from modnas.utils.logging import get_logger
from modnas.utils.config import merge_config


logger = get_logger('compat')


class VegaTrainerWrapper(TrainerBase):
    """Trainer wrapper for ModularNAS."""

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.model = trainer.model
        self.data_provider = None
        self.optimizer = None
        self.lr_scheduler = None
        self.trainer_loss = None
        self.proc_batch = None
        self.cur_batch = None
        self.step = -1
        self.trained = False
        self.built = False
        self._conditions = {}
        self._data = {}
        self._disabled = set()

    def init(self, *args, **kwargs):
        """Initialize Vega Trainer and binds its components."""
        self.trainer.build()
        self.trainer.callbacks.before_train()
        if not self.built:
            self.train_metrics = self.trainer.train_metrics
            self.valid_metrics = self.trainer.valid_metrics
        else:
            self.trainer.train_metrics = self.train_metrics
            self.trainer.valid_metrics = self.valid_metrics
        self.model = self.trainer.model
        self.optimizer = self.trainer.optimizer
        self.lr_scheduler = self.trainer.lr_scheduler
        self.trainer_loss = self.trainer_loss or self.trainer.loss
        self.proc_batch = self.proc_batch or self.trainer.make_batch
        self.wrap_make_batch()
        self.data_provider = DefaultDataProvider(self.trainer.train_loader, self.trainer.valid_loader)
        self.built = True

    def wrap_make_batch(self):
        """Wrap Trainer make_batch."""
        def make_batch(batch):
            batch = self.proc_batch(batch)
            self.cur_batch = batch
            return batch

        self.trainer.make_batch = make_batch

    def wrap_loss(self, estim):
        """Wrap Trainer loss with Estimator loss."""
        self.trainer.loss = lambda o, t, e=estim: e.loss(self.cur_batch, o, mode='train')

    def loss(self, output=None, data=None, model=None):
        """Return loss."""
        return None if self.trainer_loss is None else self.trainer_loss(output, data[-1])

    def get_num_train_batch(self, epoch):
        """Return number of train batches in current epoch."""
        return 0 if self.data_provider is None else self.data_provider.get_num_train_batch(epoch=epoch)

    def get_num_valid_batch(self, epoch):
        """Return number of validate batches in current epoch."""
        return 0 if self.data_provider is None else self.data_provider.get_num_valid_batch(epoch=epoch)

    def get_next_train_batch(self):
        """Return the next train batch."""
        return self.proc_batch(self.data_provider.get_next_train_batch())

    def get_next_valid_batch(self):
        """Return the next validate batch."""
        return self.proc_batch(self.data_provider.get_next_valid_batch())

    def notify(self, msg, data=None):
        """Notify a message."""
        logger.debug('notify: {}'.format(msg))
        if msg in self._disabled:
            return
        cond = self._conditions.get(msg, None)
        self._data[msg] = data
        if cond is None:
            self._conditions[msg] = 1
        elif isinstance(cond, int):
            return
        else:
            cond.acquire()
            cond.notifyAll()
            cond.release()
            self._conditions.pop(msg)

    def wait(self, msg):
        """Wait for a message."""
        logger.debug('wait: {}'.format(msg))
        if msg in self._disabled:
            return
        data = self._data.pop(msg, None)
        cond = self._conditions.get(msg, None)
        if isinstance(cond, int):
            self._conditions[msg] -= 1
            if not self._conditions[msg]:
                del self._conditions[msg]
            return data
        cond = threading.Condition()
        self._conditions[msg] = cond
        cond.acquire()
        cond.wait()
        return data

    def notify_all(self):
        """Notify all waiting message."""
        for msg in list(self._conditions.keys()):
            self.notify(msg)

    def disable_cond(self, msg):
        """Disable a message."""
        if msg in self._disabled:
            return
        self.notify(msg)
        self._disabled.add(msg)

    def enable_cond(self, msg):
        """Enable a message."""
        if msg not in self._disabled:
            return
        self._disabled.remove(msg)

    def get_lr(self):
        """Return current learning rate."""
        if self.lr_scheduler:
            if hasattr(self.lr_scheduler, 'get_last_lr'):
                return self.lr_scheduler.get_last_lr()[0]
            return self.lr_scheduler.get_lr()[0]
        return self.optimizer.param_groups[0]['lr']

    def get_optimizer(self):
        """Return the parameter optimizer."""
        return self.optimizer

    def train_epoch(self, *args, **kwargs):
        """Train for one epoch."""
        self.disable_cond('before_train_step')
        self.notify('before_epoch')
        self.wait('after_epoch')
        self.enable_cond('before_train_step')
        self.step = -1
        self.trained = True
        return self.train_metrics.results

    def valid_epoch(self, *args, **kwargs):
        """Validate for one epoch."""
        if not self.trained:
            self.notify('before_epoch')
        self.valid_metrics.reset()
        self.notify('before_valid')
        self.trainer._valid_epoch()
        return self.valid_metrics.results

    def train_step(self, *args, **kwargs):
        """Train for one step."""
        self.step += 1
        if not self.step:
            self.notify('before_epoch')
        self.notify('before_train_step')
        self.trained = True
        self.wait('after_train_step')
        return self.train_metrics.results

    def valid_step(self, *args, **kwargs):
        """Validate for one step."""
        self.notify('before_valid_step')
        self.wait('after_valid_step')
        return self.train_metrics.results


def _patch_fmt_config(conf, ctx):
    """Return config with formatted string."""
    if isinstance(conf, str):
        return conf.format(**ctx)
    if isinstance(conf, dict):
        conf.update({k: _patch_fmt_config(v, ctx) for k, v in conf.items()})
        return conf
    if isinstance(conf, list):
        return [_patch_fmt_config(v, ctx) for v in conf]
    return conf


@ClassFactory.register(ClassType.CALLBACK)
class ModNasTrainerCallback(Callback):
    """Trainer callback for ModularNAS."""

    disable_callbacks = ["ModelStatistics"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estim_th = None
        self.estim_ret = None
        self.initialized = False

    def init(self):
        """Initialize ModularNAS components and Vega Trainer."""
        self.config = _patch_fmt_config(self.config, {
            'local_worker_path': self.trainer.get_local_worker_path(),
            'local_base_path': self.trainer.local_base_path,
            'local_output_path': self.trainer.local_output_path,
        })
        self.config['name'] = self.config.get('name', 'default')
        self.config['routine'] = self.config.get('routine', 'search')
        self.config['expman'] = self.config.get('expman', {})
        self.config['expman']['root_dir'] = FileOps.join_path(self.trainer.get_local_worker_path(), 'exp')
        self.config = merge_config(self.config, self.model.config)
        ctx = init_all(config=self.config, base_model=None)
        self.__dict__.update(ctx)
        if self.model.net is None:
            self.model.net = list(self.estims.values())[0].model
        if self.optim:
            self.search_alg.set_optim(self.optim)
        self.wrp_trainer = VegaTrainerWrapper(self.trainer)
        self.wrp_trainer.init()

    def before_train(self, logs=None):
        """Be called before the training process."""
        if self.initialized:
            return
        self.initialized = True
        self.trainer_config = self.trainer.config
        self.config = copy.deepcopy(self.trainer_config.modnas)
        self.model = self.trainer.model
        self.search_alg = None
        if self.config.get('vega_train', False) is False:
            self.search_alg = SearchAlgorithm(SearchSpace())
        self.trainer.train_loader = self.trainer._init_dataloader(mode='train')
        self.trainer.valid_loader = self.trainer._init_dataloader(mode='val')
        self.init()
        if self.config.get('disable_estim'):
            self.wrp_trainer.disable_cond('before_epoch')
            self.wrp_trainer.disable_cond('before_train_step')
            return

        def estim_runner():
            try:
                for estim in self.estims.values():
                    estim.set_trainer(self.wrp_trainer)
                    estim.config.epochs = estim.config.get('epochs', self.trainer_config.epochs)
                results = {}
                for estim_name, estim in self.estims.items():
                    logger.info('Running estim: {} type: {}'.format(estim_name, estim.__class__.__name__))
                    self.wrp_trainer.wrap_loss(estim)
                    ret = estim.run(self.search_alg)
                    results[estim_name] = ret
                logger.info('All results: {{\n{}\n}}'.format('\n'.join(
                    ['{}: {}'.format(k, v) for k, v in results.items()])))
                results['final'] = ret
                self.estim_ret = results
            except Exception:
                logging.debug(traceback.format_exc())
            # try to release the trainer
            self.trainer.train_loader = []
            self.trainer.valid_loader = []
            self.wrp_trainer.notify_all()
            self.wrp_trainer.disable_cond('before_epoch')
            self.wrp_trainer.disable_cond('before_train_step')

        # start estim coroutine
        estim_th = threading.Thread(target=estim_runner)
        estim_th.setDaemon(True)
        estim_th.start()
        self.estim_th = estim_th

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        self.wrp_trainer.wait('before_epoch')

    def before_train_step(self, epoch, logs=None):
        """Be called before a batch training."""
        self.wrp_trainer.wait('before_train_step')

    def after_train_step(self, batch_index, logs=None):
        """Be called after each batch training."""
        if batch_index == len(self.trainer.train_loader) - 1:
            self.wrp_trainer.step = -1
        self.wrp_trainer.notify('after_train_step', {
            'batch_index': batch_index,
            'logs': logs,
        })

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        self.wrp_trainer.notify('after_epoch', {
            'epoch': epoch,
            'logs': logs,
        })

    def after_train(self, logs=None):
        """Be called after Training."""
        self.trainer._backup()
        self.wrp_trainer.notify('after_train', {
            'logs': logs,
        })
        if self.estim_th:
            self.estim_th.join()
        ret = self.estim_ret.get('final')
        self.trainer.performance = {'default': ret.get('best_score')}
        desc = self.trainer.model_desc.copy()
        desc['custom']['arch_desc'] = ret.get('best_arch_desc')
        # force update trainer record
        ReportClient().update(self.trainer.step_name, self.trainer.worker_id, desc=desc)

    def after_valid_step(self, batch_index, logs=None):
        """Be called after a batch validation."""
        self.wrp_trainer.notify('after_valid_step', {
            'batch_index': batch_index,
            'logs': logs,
        })

    def after_valid(self, logs=None):
        """Be called after the validation."""
        self.wrp_trainer.notify('after_valid', {
            'logs': logs,
        })
