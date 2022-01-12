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

"""Supernet-based Estimator."""
import itertools
from modnas.core.param_space import ParamSpace
from modnas.registry.estim import register
from ..base import EstimBase


@register
class SuperNetEstim(EstimBase):
    """Supernet-based Estimator class."""

    def step(self, params):
        """Return evaluation results of a parameter set."""
        ParamSpace().update_params(params)
        arch_desc = self.get_arch_desc()
        ret = self.compute_metrics()
        self.logger.info('Evaluate: {} -> {}'.format(arch_desc, ret))
        return ret

    def print_tensor_params(self, max_num=3):
        """Log current tensor parameter values."""
        logger = self.logger
        ap_cont = tuple(a.detach().softmax(dim=-1).cpu().numpy() for a in ParamSpace().tensor_values())
        max_num = min(len(ap_cont) // 2, max_num)
        logger.info('TENSOR: {}\n{}'.format(
            len(ap_cont), '\n'.join([str(a) for a in (ap_cont[:max_num] + ('...', ) + ap_cont[-max_num:])])))

    def run(self, optim):
        """Run Estimator routine."""
        self.reset_trainer()
        config = self.config
        tot_epochs = config.epochs
        for epoch in itertools.count(self.cur_epoch + 1):
            if (self.run_epoch(optim, epoch=epoch, tot_epochs=tot_epochs) or {}).get('stop'):
                break

    def run_epoch(self, optim, epoch, tot_epochs):
        """Run Estimator routine for one epoch."""
        if epoch == tot_epochs:
            return {'stop': True}
        config = self.config
        # train
        self.print_tensor_params()
        n_trn_batch = self.get_num_train_batch(epoch)
        n_val_batch = self.get_num_valid_batch(epoch)
        update_arch = False
        arch_epoch_start = config.arch_update_epoch_start
        arch_epoch_intv = config.arch_update_epoch_intv
        if epoch >= arch_epoch_start and (epoch - arch_epoch_start) % arch_epoch_intv == 0:
            update_arch = True
            arch_update_intv = config.arch_update_intv
            if arch_update_intv == -1:  # update proportionally
                arch_update_intv = max(n_trn_batch / n_val_batch, 1) if n_val_batch else 1
            elif arch_update_intv == 0:  # update last step
                arch_update_intv = n_trn_batch
            arch_update_batch = config.arch_update_batch
        arch_step = 0
        for step in range(n_trn_batch):
            # optim step
            if update_arch and (step + 1) // arch_update_intv > arch_step:
                for _ in range(arch_update_batch):
                    optim.step(self)
                arch_step += 1
            # supernet step
            optim.next()
            self.trainer.train_step(estim=self,
                                    model=self.model,
                                    epoch=epoch,
                                    tot_epochs=tot_epochs,
                                    step=step,
                                    tot_steps=n_trn_batch)
        # eval
        self.clear_buffer()
        self.stepped(dict(ParamSpace().named_param_values()))
        self.wait_done()
