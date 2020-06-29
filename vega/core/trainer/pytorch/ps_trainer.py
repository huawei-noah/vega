# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Parameter Sharing Trainer."""
import os
import json
import numpy as np
import logging
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from vega.core.common import FileOps, Config, DefaultConfig
from vega.core.trainer.pytorch.trainer import Trainer
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space import SearchSpace
from vega.search_space.search_algs import SearchAlgorithm
from vega.datasets.pytorch.common.dataset import Dataset
from vega.core.metrics.pytorch.metrics import Metrics


@ClassFactory.register(ClassType.TRAINER)
class ParameterSharingTrainer(Trainer):
    """Parameter Sharing Trainer Class.

    :param model: network model
    :type model: torch.nn.Module
    :param id: trainer id
    :type id: int
    """

    def __init__(self, model, id):
        super(ParameterSharingTrainer, self).__init__(model, id)
        self.device = self.cfg.device

    def _train_model_step(self, input, target):
        """Train model in one step.

        :param input: input data
        :type input: torch tensor
        :param target: target label data
        :type target: torch tensor
        """
        self.optimizer.zero_grad()
        logits = self.model(input)
        loss = self.loss_fn(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return logits

    def _train_arch_step(self, train_input, train_target, valid_input, valid_target):
        """Abstract function of training arch in one step.

        :param train_input: train input data
        :param train_target: train target data
        :param valid_input: valid input data
        :param valid_target: valid target data
        """
        raise NotImplementedError

    def _train(self, model):
        """Train Parameter Sharing model with train and valid data.

        :param model: parameter sharing super model
        :type model: torch.nn.Module
        """
        metrics = Metrics(self.cfg.metric)
        model.train()
        valid_loader_iter = iter(self.valid_loader)
        step = 0
        for (train_input, train_target) in self.train_loader:
            try:
                valid_input, valid_target = next(valid_loader_iter)
            except Exception:
                valid_loader_iter = iter(self.valid_loader)
                valid_input, valid_target = next(valid_loader_iter)
            train_input, train_target = train_input.to(self.device), train_target.to(self.device)
            valid_input, valid_target = valid_input.to(self.device), valid_target.to(self.device)
            self._train_arch_step(train_input, train_target, valid_input, valid_target)
            train_logits = self._train_model_step(train_input, train_target)
            metrics(train_logits, train_target)
            top1 = metrics.results[0]
            if self._first_rank and step % self.cfg.print_step_interval == 0:
                logging.info("step [{}/{}], top1: {}".format(step + 1, len(self.train_loader), top1))
            step += 1

    def _valid(self, model, loader):
        """Validate Parameter Sharing model with data.

        :param model: network model
        :type model: torch.nn.Module
        :param loader: data loader
        :type loader: DataLoader
        :return: top1, top5
        :rtype: float, float
        """
        metrics = Metrics(self.cfg.metric)
        model.eval()
        with torch.no_grad():
            for _, (input, target) in enumerate(loader):
                input, target = input.to(self.device), target.to(self.device)
                logits = model(input)
                metrics(logits, target)
        top1 = metrics.results[0]
        top5 = metrics.results[1]
        return top1, top5

    def train_process(self):
        """Train process of parameter sharing."""
        self.train_loader = Dataset(mode='train').dataloader
        self.valid_loader = Dataset(mode='val').dataloader
        self.model = self.model.to(self.device)
        self.search_alg = SearchAlgorithm(SearchSpace())
        self.set_algorithm_model(self.model)
        self.optimizer = self._init_optimizer()
        self.lr_scheduler = self._init_lr_scheduler()
        self.loss_fn = self._init_loss()
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        for i in range(self.cfg.epochs):
            self._train(self.model)
            train_top1, train_top5 = self._valid(self.model, self.train_loader)
            valid_top1, valid_top5 = self._valid(self.model, self.valid_loader)
            self.lr_scheduler.step()
            child_desc_temp = self.search_alg.codec.calc_genotype(self.model.arch_weights)
            logging.info(F.softmax(self.model.alphas_normal, dim=-1))
            logging.info(F.softmax(self.model.alphas_reduce, dim=-1))
            logging.info('normal = %s', child_desc_temp[0])
            logging.info('reduce = %s', child_desc_temp[1])
            logging.info('Epoch {}: train top1: {}, train top5: {}'.format(i, train_top1, train_top5))
            logging.info('Epoch {}: valid top1: {}, valid top5: {}'.format(i, valid_top1, valid_top5))
        child_desc = self.search_alg.codec.decode(self.model.arch_weights)
        self._save_descript(child_desc)

    def _save_descript(self, descript):
        """Save result descript.

        :param descript: darts search result descript
        :type descript: dict or Config
        """
        template_file = self.cfg.darts_template_file
        genotypes = self.search_alg.codec.calc_genotype(self.model.arch_weights)
        if template_file == "{default_darts_cifar10_template}":
            template = DefaultConfig().data.default_darts_cifar10_template
        elif template_file == "{default_darts_imagenet_template}":
            template = DefaultConfig().data.default_darts_imagenet_template
        else:
            dst = FileOps.join_path(self.get_local_worker_path(), os.path.basename(template_file))
            FileOps.copy_file(template_file, dst)
            template = Config(dst)
        model_desc = self._gen_model_desc(genotypes, template)
        self.output_model_desc(self.worker_id, model_desc)

    def _gen_model_desc(self, genotypes, template):
        model_desc = deepcopy(template)
        model_desc.super_network.normal.genotype = genotypes[0]
        model_desc.super_network.reduce.genotype = genotypes[1]
        return model_desc

    def set_algorithm_model(self, model):
        """Set model to algorithm, not implemented yes."""
        pass
