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
"""AutoGate Grda version Stage1 TrainerCallback."""

import logging
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.algorithms.nas.fis.ctr_trainer_callback import CtrTrainerCallback
import torch.optim as optim
from vega.algorithms.nas.fis.grda import gRDA

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class AutoGateGrdaS1TrainerCallback(CtrTrainerCallback):
    """AutoGateGrdaS1TrainerCallback module."""

    def __init__(self):
        """Construct AutoGateGrdaS1TrainerCallback class."""
        super(CtrTrainerCallback, self).__init__()
        self.best_score = 0

        logging.info("init autogate s1 trainer callback")

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.config = self.trainer.config
        all_parameters = self.trainer.model.parameters()
        structure_params = self.trainer.model.structure_params
        net_params = [i for i in all_parameters if i not in structure_params]
        self.struc_optimizer = self._init_structure_optimizer(structure_params)
        self.net_optimizer = self._init_network_optimizer(net_params)

    def _init_structure_optimizer(self, learnable_params):
        """
        Init structure optimizer for optimize structure params in autogate model.

        :param learnable_params:  learnable structure params
        :type learnable_params: list object
        :return: optimizer object
        :rtype: torch.optim.Optimizer
        """
        logging.info("init net optimizer, lr = {}".format(self.config.struc_optim.struct_lr))
        optimizer = gRDA(learnable_params, lr=float(self.config.struc_optim.struct_lr),
                         c=float(self.config.c), mu=float(self.config.mu))

        logging.info("init structure optimizer finish.")
        return optimizer

    def _init_network_optimizer(self, learnable_params):
        """
        Init structure optimizer for optimize structure params in autogate model.

        :param learnable_params:  learnable network params
        :type learnable_params: list object
        :return: optimizer object
        :rtype: torch.optim.Optimizer
        """
        logging.info("init net optimizer, lr = {}".format(self.config.net_optim.net_lr))
        optimizer = optim.Adam(learnable_params, lr=float(self.config.net_optim.net_lr))

        logging.info("init structure optimizer finish.")
        return optimizer

    def train_step(self, batch):
        """
        Training progress for a batch data train net_param and struct_param step by step (iteratly).

        :param batch: batch train data.
        :type batch: list object
        :return: loss & training loss
        :rtype: dict object
        """
        self.trainer.model.train()
        input, target = batch

        self.net_optimizer.zero_grad()
        self.struc_optimizer.zero_grad()

        output = self.trainer.model(input)

        loss = self.trainer.loss(output, target)
        loss.backward()
        self.net_optimizer.step()
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

        output = self.trainer.model(input)
        return {'valid_batch_output': output}

    def after_valid(self, logs=None):
        """Call after_valid of the managed callbacks."""
        self.model = self.trainer.model
        feature_interaction_score = self.model.get_feature_interaction_score()
        print('get feature_interaction_score', feature_interaction_score)
        feature_interaction = []
        for feature in feature_interaction_score:
            if abs(feature_interaction_score[feature]) > 0:
                feature_interaction.append(feature)
        print('get feature_interaction', feature_interaction)

        curr_auc = float(self.trainer.valid_metrics.results['auc'])
        if curr_auc > self.best_score:
            best_config = {
                'score': curr_auc,
                'feature_interaction': feature_interaction
            }

            logging.info("BEST CONFIG IS\n{}".format(best_config))
            pickle_result_file = FileOps.join_path(
                self.trainer.local_output_path, 'best_config.pickle')
            logging.info("Saved to {}".format(pickle_result_file))
            FileOps.dump_pickle(best_config, pickle_result_file)

            self.best_score = curr_auc
