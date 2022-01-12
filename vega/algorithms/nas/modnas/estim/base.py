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

"""Base Estimator."""
import traceback
import threading
from modnas import backend
from modnas.metrics import build_metrics_all
from modnas.registry.export import build as build_exporter
from modnas.core.event import event_hooked_subclass
from modnas.utils.logging import get_logger
from modnas.registry import streamline_spec
from vega.common import FileOps


def build_criterions_all(crit_configs, device_ids=None):
    """Build Criterions from configs."""
    crits_all = []
    crits_train = []
    crits_eval = []
    crits_valid = []
    for crit_conf in streamline_spec(crit_configs):
        crit = backend.get_criterion(crit_conf, device_ids=device_ids)
        crit_mode = crit_conf['mode'] if isinstance(crit_conf, dict) and 'mode' in crit_conf else 'all'
        if not isinstance(crit_mode, list):
            crit_mode = [crit_mode]
        if 'all' in crit_mode:
            crits_all.append(crit)
        if 'train' in crit_mode:
            crits_train.append(crit)
        if 'eval' in crit_mode:
            crits_eval.append(crit)
        if 'valid' in crit_mode:
            crits_valid.append(crit)
    return crits_all, crits_train, crits_eval, crits_valid


@event_hooked_subclass
class EstimBase():
    """Base Estimator class."""

    logger = get_logger('estim')

    def __init__(self,
                 config=None,
                 expman=None,
                 trainer=None,
                 constructor=None,
                 exporter=None,
                 model=None,
                 writer=None,
                 name=None):
        self.name = '' if name is None else name
        self.config = config
        self.expman = expman
        self.constructor = constructor
        self.exporter = exporter
        self.model = model
        self.writer = writer
        self.cur_epoch = -1
        self.metrics = build_metrics_all(config.get('metrics', None), self)
        self.criterions_all, self.criterions_train, self.criterions_eval, self.criterions_valid = build_criterions_all(
            config.get('criterion', None), getattr(model, 'device_ids', None))
        self.trainer = trainer
        self._results = []
        self._inputs = []
        self._arch_descs = []
        self._step_cond = threading.Lock()
        self._n_step_waiting = 0
        self._cur_trn_batch = None
        self._cur_val_batch = None

    def set_trainer(self, trainer):
        """Set current trainer."""
        self.trainer = trainer

    def model_output(self, *args, data=None, model=None, **kwargs):
        """Return model output for given data."""
        model = self.model if model is None else model
        return self.trainer.model_output(*args, data=data, model=model, **kwargs)

    def loss(self, data, output=None, model=None, mode=None):
        """Return loss."""
        model = self.model if model is None else model
        output = self.model_output(data=data, model=model) if output is None else output
        if mode is None:
            crits = []
        elif mode == 'train':
            crits = self.criterions_train
        elif mode == 'eval':
            crits = self.criterions_eval
        elif mode == 'valid':
            crits = self.criterions_valid
        else:
            raise ValueError('invalid criterion mode: {}'.format(mode))
        crits = self.criterions_all + crits
        loss = self.trainer.loss(model=model, output=output, data=data)
        for crit in crits:
            loss = crit(loss, self, output, *data)
        return loss

    def loss_output(self, data, model=None, mode=None):
        """Return loss and model output."""
        model = self.model if model is None else model
        output = self.model_output(data=data, model=model)
        return self.loss(data, output, model, mode), output

    def step(self, params):
        """Return evaluation results of a parameter set."""
        raise NotImplementedError

    def stepped(self, params):
        """Return evaluation results of a parameter set."""
        if not self._step_cond.locked():
            self._step_cond.acquire()
        self._n_step_waiting += 1
        value = self.step(params)
        if value is not None:
            self.step_done(params, value)

    def wait_done(self):
        """Wait evaluation steps to finish."""
        self._step_cond.acquire()
        self._step_cond.release()

    def step_done(self, params, value, arch_desc=None):
        """Store evaluation results of a parameter set."""
        self._inputs.append(params)
        self._results.append(value)
        self._arch_descs.append(self.get_arch_desc() if arch_desc is None else arch_desc)
        self._n_step_waiting -= 1
        if self._n_step_waiting == 0:
            self._step_cond.release()

    def print_model_info(self):
        """Output model information."""
        model = self.model
        if model is not None:
            self.logger.info(backend.model_summary(model))

    def clear_buffer(self):
        """Clear evaluation results."""
        self._inputs, self._results, self._arch_descs = [], [], []

    def get_last_results(self):
        """Return last evaluation results."""
        return self._inputs, self._results

    def buffer(self):
        """Return generator over evaluated results with parameters and arch_descs."""
        for inp, res, desc in zip(self._inputs, self._results, self._arch_descs):
            yield inp, res, desc

    def compute_metrics(self, *args, name=None, model=None, to_scalar=True, **kwargs):
        """Return Metrics results."""
        def fmt_key(n, k):
            return '{}.{}'.format(n, k)

        def flatten_dict(n, r):
            if isinstance(r, dict):
                return {fmt_key(n, k): flatten_dict(fmt_key(n, k), v) for k, v in r.items()}
            return r

        def merge_results(dct, n, r):
            if not isinstance(r, dict):
                r = {n: r}
            r = {k: None if v is None else (float(v) if to_scalar else v) for k, v in r.items()}
            dct.update(r)

        ret = {}
        model = self.model if model is None else model
        names = [name] if name is not None else self.metrics.keys()
        for mt_name in names:
            res = self.metrics[mt_name](model, *args, **kwargs)
            merge_results(ret, mt_name, flatten_dict(mt_name, res))
        return ret

    def run_epoch(self, optim, epoch, tot_epochs):
        """Run Estimator routine for one epoch."""
        raise NotImplementedError

    def run(self, optim):
        """Run Estimator routine."""
        raise NotImplementedError

    def get_score(self, res):
        """Return scalar value from evaluation results."""
        if not isinstance(res, dict):
            return res
        score = res.get('default', None)
        if score is None:
            score = 0 if len(res) == 0 else list(res.values())[0]
        return score

    def train_epoch(self, epoch, tot_epochs, model=None):
        """Train model for one epoch."""
        model = self.model if model is None else model
        ret = self.trainer.train_epoch(estim=self,
                                       model=model,
                                       tot_steps=self.get_num_train_batch(epoch),
                                       epoch=epoch,
                                       tot_epochs=tot_epochs)
        return ret

    def train_step(self, epoch, tot_epochs, step, tot_steps, model=None):
        """Train model for one step."""
        model = self.model if model is None else model
        return self.trainer.train_step(estim=self,
                                       model=model,
                                       epoch=epoch,
                                       tot_epochs=tot_epochs,
                                       step=step,
                                       tot_steps=tot_steps)

    def valid_epoch(self, epoch=0, tot_epochs=1, model=None):
        """Validate model for one epoch."""
        model = self.model if model is None else model
        return self.trainer.valid_epoch(estim=self,
                                        model=model,
                                        tot_steps=self.get_num_valid_batch(epoch),
                                        epoch=epoch,
                                        tot_epochs=tot_epochs)

    def valid_step(self, epoch, tot_epochs, step, tot_steps, model=None):
        """Validate model for one step."""
        model = self.model if model is None else model
        return self.trainer.valid_step(estim=self,
                                       model=model,
                                       epoch=epoch,
                                       tot_epochs=tot_epochs,
                                       step=step,
                                       tot_steps=tot_steps)

    def reset_trainer(self, *args, trainer_config=None, model=None, **kwargs):
        """Reinitialize trainer."""
        model = self.model if model is None else model
        trainer_config = trainer_config or {}
        trainer_config.update({
            'epochs': self.config.epochs
        })
        trainer_config.update(kwargs)
        if self.trainer is not None:
            self.trainer.init(*args, model=model, config=trainer_config)
        self.cur_epoch = -1

    def get_num_train_batch(self, epoch=None):
        """Return number of training batches."""
        epoch = self.cur_epoch if epoch is None else epoch
        return 0 if self.trainer is None else self.trainer.get_num_train_batch(epoch=epoch)

    def get_num_valid_batch(self, epoch=None):
        """Return number of validating batches."""
        epoch = self.cur_epoch if epoch is None else epoch
        return 0 if self.trainer is None else self.trainer.get_num_valid_batch(epoch=epoch)

    def get_next_train_batch(self):
        """Return the next training batch."""
        ret = self.trainer.get_next_train_batch()
        self._cur_trn_batch = ret
        return ret

    def get_cur_train_batch(self):
        """Return the current training batch."""
        return self._cur_trn_batch or self.get_next_train_batch()

    def get_next_valid_batch(self):
        """Return the next validating batch."""
        ret = self.trainer.get_next_valid_batch()
        self._cur_val_batch = ret
        return ret

    def get_cur_valid_batch(self):
        """Return the current validating batch."""
        return self._cur_val_batch

    def load_state_dict(self, state_dict):
        """Resume states."""
        pass

    def state_dict(self):
        """Return current states."""
        return {'cur_epoch': self.cur_epoch}

    def get_arch_desc(self):
        """Return current archdesc."""
        return None if self.exporter is None else self.exporter(self.model)

    def save_model(self, save_name=None, exporter='DefaultTorchCheckpointExporter'):
        """Save model checkpoint to file."""
        if self.model is None:
            return
        expman = self.expman
        save_name = 'model_{}_{}.pt'.format(self.name, save_name)
        chkpt_path = expman.join('chkpt', save_name)
        build_exporter(exporter, path=chkpt_path)(self.model)

    def save(self, epoch=None, save_name=None):
        """Save Estimator states to file."""
        expman = self.expman
        logger = self.logger
        save_name = 'estim_{}_{}.pkl'.format(self.name, save_name)
        chkpt_path = expman.join('chkpt', save_name)
        epoch = epoch or self.cur_epoch
        try:
            chkpt = self.state_dict()
            FileOps.dump_pickle(chkpt, chkpt_path)
        except RuntimeError as e:
            logger.debug(traceback.format_exc())
            logger.error(f"Failed saving estimator: {e}")

    def save_checkpoint(self, epoch=None, save_name=None):
        """Save Estimator & model to file."""
        epoch = epoch or self.cur_epoch
        save_name = save_name or 'ep{:03d}'.format(epoch + 1)
        self.save_model(save_name)
        self.save(epoch, save_name)

    def save_arch_desc(self, epoch=None, arch_desc=None, save_name=None, exporter='DefaultToFileExporter'):
        """Save archdesc to file."""
        expman = self.expman
        logger = self.logger
        if save_name is not None:
            fname = '{}_{}'.format(self.name, save_name)
        else:
            epoch = epoch or self.cur_epoch
            fname = '{}_ep{:03d}'.format(self.name, epoch + 1)
        save_path = expman.join('output', fname)
        try:
            build_exporter(exporter, path=save_path)(arch_desc)
        except RuntimeError as e:
            logger.debug(traceback.format_exc())
            logger.error(f"Failed saving arch_desc, message: {e}")

    def load(self, chkpt_path):
        """Load states from file."""
        if chkpt_path is None:
            return
        self.logger.info("Resuming from checkpoint: {}".format(chkpt_path))
        chkpt = FileOps.load_pickle(chkpt_path)
        self.load_state_dict(chkpt)
