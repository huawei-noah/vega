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

"""SGAS trainer."""

import logging
import vega
import torch
import torch.nn.functional as F
from vega.algorithms.nas.darts_cnn import DartsTrainerCallback
from vega.common import ClassFactory, ClassType
from vega.core.search_space import SearchSpace
from vega.core.search_algs import SearchAlgorithm


@ClassFactory.register(ClassType.CALLBACK)
class SGASTrainerCallback(DartsTrainerCallback):
    """A special callback for DartsTrainer."""

    disable_callbacks = ["ModelCheckpoint"]

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.config = self.trainer.config
        self.unrolled = self.trainer.config.unrolled
        self.device = self.trainer.config.device
        self.model = self.trainer.model
        self.optimizer = self.trainer.optimizer
        self.lr_scheduler = self.trainer.lr_scheduler
        self.loss = self.trainer.loss
        self.search_alg = SearchAlgorithm(SearchSpace())
        self._set_algorithm_model(self.model)
        self.trainer.train_loader = self.trainer._init_dataloader(mode='train')
        self.trainer.valid_loader = self.trainer._init_dataloader(mode='val')
        normal_selected_idxs = torch.tensor(len(self.model.alphas_normal) * [-1],
                                            requires_grad=False, dtype=torch.int).cuda()
        reduce_selected_idxs = torch.tensor(len(self.model.alphas_reduce) * [-1],
                                            requires_grad=False, dtype=torch.int).cuda()
        normal_candidate_flags = torch.tensor(len(self.model.alphas_normal) * [True],
                                              requires_grad=False, dtype=torch.bool).cuda()
        reduce_candidate_flags = torch.tensor(len(self.model.alphas_reduce) * [True],
                                              requires_grad=False, dtype=torch.bool).cuda()
        logging.info('normal_selected_idxs: {}'.format(normal_selected_idxs))
        logging.info('reduce_selected_idxs: {}'.format(reduce_selected_idxs))
        logging.info('normal_candidate_flags: {}'.format(normal_candidate_flags))
        logging.info('reduce_candidate_flags: {}'.format(reduce_candidate_flags))
        self.model.normal_selected_idxs = normal_selected_idxs
        self.model.reduce_selected_idxs = reduce_selected_idxs
        self.model.normal_candidate_flags = normal_candidate_flags
        self.model.reduce_candidate_flags = reduce_candidate_flags
        logging.info(F.softmax(torch.stack(self.model.alphas_normal, dim=0), dim=-1).detach())
        logging.info(F.softmax(torch.stack(self.model.alphas_reduce, dim=0), dim=-1).detach())
        self.normal_probs_history = []
        self.reduce_probs_history = []

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        if vega.is_torch_backend():
            self.valid_loader_iter = iter(self.trainer.valid_loader)

    def before_train_step(self, epoch, logs=None):
        """Be called before a batch training."""
        # Get current train batch directly from logs
        train_batch = logs['train_batch']
        train_input, train_target = train_batch
        # Prepare valid batch data by using valid loader from trainer
        try:
            valid_input, valid_target = next(self.valid_loader_iter)
        except Exception:
            self.valid_loader_iter = iter(self.trainer.valid_loader)
            valid_input, valid_target = next(self.valid_loader_iter)
        valid_input, valid_target = valid_input.to(self.device), valid_target.to(self.device)
        # Call arch search step
        self._train_arch_step(train_input, train_target, valid_input, valid_target)

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        child_desc_temp = self.search_alg.codec.calc_genotype(self._get_arch_weights())
        logging.info('normal = %s', child_desc_temp[0])
        logging.info('reduce = %s', child_desc_temp[1])
        normal_edge_decision = self.search_alg.edge_decision('normal',
                                                             self.model.alphas_normal,
                                                             self.model.normal_selected_idxs,
                                                             self.model.normal_candidate_flags,
                                                             self.normal_probs_history,
                                                             epoch)
        saved_memory_normal, self.model.normal_selected_idxs, self.model.normal_candidate_flags = normal_edge_decision
        reduce_edge_decision = self.search_alg.edge_decision('reduce',
                                                             self.model.alphas_reduce,
                                                             self.model.reduce_selected_idxs,
                                                             self.model.reduce_candidate_flags,
                                                             self.reduce_probs_history,
                                                             epoch)
        saved_memory_reduce, self.model.reduce_selected_idxs, self.model.reduce_candidate_flags = reduce_edge_decision
        if saved_memory_normal or saved_memory_reduce:
            torch.cuda.empty_cache()
        self._save_descript()
