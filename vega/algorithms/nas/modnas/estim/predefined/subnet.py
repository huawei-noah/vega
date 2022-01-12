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

"""Subnet-based Estimator."""
import itertools
import traceback
from modnas import backend
from modnas.core.param_space import ParamSpace
from modnas.registry.estim import register
from ..base import EstimBase


@register
class SubNetEstim(EstimBase):
    """Subnet-based Estimator class."""

    def __init__(self, rebuild_subnet=False, num_bn_batch=100, clear_subnet_bn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebuild_subnet = rebuild_subnet
        self.num_bn_batch = num_bn_batch
        self.clear_subnet_bn = clear_subnet_bn

    def step(self, params):
        """Return evaluation results of a parameter set."""
        ParamSpace().update_params(params)
        arch_desc = self.get_arch_desc()
        config = self.config
        try:
            self.construct_subnet(arch_desc)
        except RuntimeError as e:
            self.logger.debug(traceback.format_exc())
            self.logger.info(f'subnet construct failed, message: {e}')
            ret = {'error_no': -1}
            return ret
        tot_epochs = config.subnet_epochs
        if tot_epochs > 0:
            self.reset_trainer(epochs=tot_epochs)
            for epoch in itertools.count(0):
                if epoch == tot_epochs:
                    break
                # train
                self.train_epoch(epoch=epoch, tot_epochs=tot_epochs)
        ret = self.compute_metrics()
        self.logger.info('Evaluate: {} -> {}'.format(arch_desc, ret))
        return ret

    def construct_subnet(self, arch_desc):
        """Return subnet built from archdesc."""
        if self.rebuild_subnet:
            self.model = self.constructor(arch_desc=arch_desc)
        else:
            backend.recompute_bn_running_statistics(self.model, self.trainer, self.num_bn_batch, self.clear_subnet_bn)

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
        self.reset_trainer()
        config = self.config
        tot_epochs = config.epochs
        for epoch in itertools.count(self.cur_epoch + 1):
            if (self.run_epoch(optim, epoch=epoch, tot_epochs=tot_epochs) or {}).get('stop'):
                break
