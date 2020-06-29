# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""DARTS trainer."""
import os
import logging
from copy import deepcopy
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import FileOps, Config, DefaultConfig
from vega.search_space import SearchSpace
from vega.search_space.search_algs import SearchAlgorithm
from vega.core.trainer.callbacks import Callback


@ClassFactory.register(ClassType.CALLBACK)
class DartsTrainerCallback(Callback):
    """A special callback for DartsTrainer."""

    def before_train(self, epoch, logs=None):
        """Be called before the training process."""
        self.cfg = self.trainer.cfg
        self.trainer.auto_save_ckpt = False
        self.trainer.auto_save_perf = False
        self.unrolled = self.trainer.cfg.get('unrolled', True)
        self.device = self.trainer.cfg.get('device', 0)
        self.model = self.trainer.model
        self.optimizer = self.trainer.optimizer
        self.lr_scheduler = self.trainer.lr_scheduler
        self.loss = self.trainer.loss
        self.search_alg = SearchAlgorithm(SearchSpace())
        self._set_algorithm_model(self.model)
        self.trainer.train_loader = self.trainer._init_dataloader(mode='train')
        self.trainer.valid_loader = self.trainer._init_dataloader(mode='val')

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        self.valid_loader_iter = iter(self.trainer.valid_loader)

    def before_train_step(self, epoch, logs=None):
        """Be called before a batch training."""
        # Get curretn train batch directly from logs
        train_batch = logs['train_batch']
        train_input, train_target = train_batch
        # Prepare valid batch data by using valid loader from trianer
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
        child_desc_temp = self.search_alg.codec.calc_genotype(self.model.arch_weights)
        logging.info('normal = %s', child_desc_temp[0])
        logging.info('reduce = %s', child_desc_temp[1])
        logging.info('lr = {}'.format(self.lr_scheduler.get_lr()[0]))

    def after_train(self, epoch, logs=None):
        """Be called after Training."""
        child_desc = self.search_alg.codec.decode(self.model.arch_weights)
        self._save_descript(child_desc)
        self.trainer._backup()

    def _train_arch_step(self, train_input, train_target, valid_input, valid_target):
        lr = self.lr_scheduler.get_lr()[0]
        self.search_alg.step(train_input, train_target, valid_input, valid_target,
                             lr, self.optimizer, self.loss, self.unrolled)

    def _set_algorithm_model(self, model):
        self.search_alg.set_model(model)

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
            dst = FileOps.join_path(self.trainer.get_local_worker_path(), os.path.basename(template_file))
            FileOps.copy_file(template_file, dst)
            template = Config(dst)
        model_desc = self._gen_model_desc(genotypes, template)
        self.trainer.output_model_desc(self.trainer.worker_id, model_desc)

    def _gen_model_desc(self, genotypes, template):
        model_desc = deepcopy(template)
        model_desc.super_network.normal.genotype = genotypes[0]
        model_desc.super_network.reduce.genotype = genotypes[1]
        return model_desc
