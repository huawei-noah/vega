# -*- coding: utf-8 -*-

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

"""AutoGroup algorithm trainer callback file."""

import logging
import torch.optim as optim
from vega.common import ClassFactory, ClassType
from .ctr_trainer_callback import CtrTrainerCallback

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class AutoGroupTrainerCallback(CtrTrainerCallback):
    """AutoGroup algorithm trainer callbacks.

    Different from other trainer method, AutoGroup respectively train network params and structure params,
    thus, there define two optimizers to train these params respectively.
    """

    def __init__(self):
        """Class of AutoGroupTrainerCallback."""
        super(AutoGroupTrainerCallback, self).__init__()
        logging.info("init autogroup trainer callback finish.")

    def before_train(self, logs=None):
        """Be called before the training process."""
        self._init_all_settings()

    def _init_all_settings(self):
        """Init all settings from config."""
        self.config = self.trainer.config
        logging.info("AutoGroupTrainerCallbacks: {}".format(self.config))
        self.struc_optimizer = self._init_structure_optimizer(self.trainer.model)
        self.net_optimizer = self._init_network_optimizer(self.trainer.model)

    def _init_structure_optimizer(self, model):
        """
        Init structure optimizer for optimize structure params in AutoGroup model.

        :param model:  Autogroup model
        :type model: torch.nn.Module
        :return: optimizer object
        :rtype: torch.optim.Optimizer
        """
        learnable_params = model.structure_params
        logging.info("init net optimizer, lr = {}".format(self.config.struc_optim.struct_lr))
        optimizer = optim.Adam(learnable_params, lr=float(self.config.struc_optim.struct_lr))

        logging.info("init structure optimizer finish.")
        return optimizer

    def _init_network_optimizer(self, model):
        """
        Init structure optimizer for optimize structure params in AutoGroup model.

        :param model:  Autogroup model
        :type model: torch.nn.Module
        :return: optimizer object
        :rtype: torch.optim.Optimizer
        """
        learnable_params = model.net_params
        optimizer = optim.Adam(learnable_params, lr=float(self.config.net_optim.net_lr))
        logging.info("init net optimizer, lr = {}".format(self.config.net_optim.net_lr))
        logging.info("init structure optimizer finish.")
        return optimizer

    def train_step(self, batch):
        """
        Training progress for a batch data.

        :param batch: batch train data.
        :type batch: list object
        :return: loss & training loss
        :rtype: dict object
        """
        self.trainer.model.train()
        input, target = batch

        # first step: train network params.
        self.net_optimizer.zero_grad()
        output = self.trainer.model(input, fix_structure=True)
        loss = self.trainer.loss(output, target)
        loss.backward()
        self.net_optimizer.step()

        # second step : train struture params
        self.struc_optimizer.zero_grad()
        struct_output = self.trainer.model(input, fix_structure=False)
        struct_loss = self.trainer.loss(struct_output, target)
        struct_loss.backward()
        self.struc_optimizer.step()

        return {'loss': loss.item(),
                'train_batch_output': output,
                'lr': self.trainer.lr_scheduler.get_lr()}

    def valid_step(self, batch):
        """
        Validate progress for a batch data.

        :param batch: batch data
        :type object
        :return: valid batch output
        :rtype: dict object
        """
        input, target = batch

        output = self.trainer.model(input, fix_structure=True)
        return {'valid_batch_output': output}
