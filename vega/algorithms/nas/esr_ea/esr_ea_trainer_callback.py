# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The trainer program for ESR_EA."""
import os
import math
import torch
import logging
import numpy as np
import pandas as pd
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space import SearchSpace
from vega.search_space.codec import Codec
from vega.search_space.networks import NetworkDesc
from vega.core.common import FileOps, Config
from .esr_ea_individual import ESRIndividual
from vega.core.trainer.callbacks import Callback


@ClassFactory.register(ClassType.CALLBACK)
class ESRTrainerCallback(Callback):
    """Construct the trainer of ESR-EA."""

    def before_train(self, epoch, logs=None):
        """Be called before the training process."""
        self.cfg = self.trainer.cfg
        # Use own save checkpoint and save performance function
        self.trainer.auto_save_ckpt = False
        self.trainer.auto_save_perf = False
        # This part is tricky and
        model = ClassFactory.__configs__.get('model', None)
        if model:
            self.model_desc = model.get("model_desc", None)
            if self.model_desc is not None:
                model = self._init_model()
                self.trainer.build(model=model)

    def make_batch(self, batch):
        """Make batch for each training step."""
        input = batch["LR"]
        target = batch["HR"]
        if self.cfg.cuda:
            input = input.cuda()
            target = target.cuda()
        return input, target

    def after_epoch(self, epoch, logs=None):
        """Be called after one epoch training."""
        # Get summary perfs from logs from built-in MetricsEvaluator callback.
        self.performance = logs.get('summary_perfs', None)
        best_valid_perfs = self.performance['best_valid_perfs']
        best_valid = list(best_valid_perfs.values())[0]
        best_changed = self.performance['best_valid_perfs_changed']
        if best_changed:
            self._save_checkpoint({"Best PSNR": best_valid, "Epoch": epoch})

    def after_train(self, logs=None):
        """Be called after the whole train process."""
        # Extract performance logs. This can be moved into builtin callback
        # if we can unify the performace content
        best_valid_perfs = self.performance['best_valid_perfs']
        best_valid = list(best_valid_perfs.values())[0]
        self._save_performance(best_valid)

    def _save_checkpoint(self, performance=None, model_name="best.pth"):
        """Save the trained model.

        :param performance: dict of all the result needed
        :type performance: dictionary
        :param model_name: name of the result file
        :type model_name: string
        :return: the path of the saved file
        :rtype: string
        """
        local_worker_path = self.trainer.get_local_worker_path()
        model_save_path = os.path.join(local_worker_path, model_name)
        torch.save({
            'model_state_dict': self.trainer.model.state_dict(),
            **performance
        }, model_save_path)

        torch.save(self.trainer.model.state_dict(), model_save_path)
        logging.info("model saved to {}".format(model_save_path))
        return model_save_path

    def _save_performance(self, performance, model_desc=None):
        """Save result of the model, and calculate pareto front.

        :param performance: The dict that contains all the result needed
        :type performance: dictionary
        :param model_desc: config of the model
        :type model_desc: dictionary
        """
        self.trainer._save_performance(performance)
        # FileOps.copy_file(self.performance_file, self.best_model_pfm)
        pd_path = os.path.join(self.trainer.local_output_path, 'population_fitness.csv')
        df = pd.DataFrame([[performance]], columns=["PSNR"])
        if not os.path.exists(pd_path):
            with open(pd_path, "w") as file:
                df.to_csv(file, index=False)
        else:
            with open(pd_path, "a") as file:
                df.to_csv(file, index=False, header=False)

    def _init_model(self):
        """Initialize the model architecture for full train step.

        :return: train model
        :rtype: class
        """
        search_space = Config({"search_space": self.model_desc})
        self.codec = Codec(self.cfg.codec, search_space)
        self.get_selected_arch()
        indiv_cfg = self.codec.decode(self.elitism)
        self.trainer.model_desc = self.elitism.active_net_list()
        # self.output_model_desc()
        net_desc = NetworkDesc(indiv_cfg)
        model = net_desc.to_model()
        return model

    def get_selected_arch(self):
        """Get the gene code of selected model architecture."""
        self.elitism = ESRIndividual(self.codec, self.cfg)
        if "model_arch" in self.cfg and self.cfg.model_arch is not None:
            self.elitism.update_gene(self.cfg.model_arch)
        else:
            sel_arch_file = self.cfg.model_desc_file
            sel_arch = np.load(sel_arch_file)
            self.elitism.update_gene(sel_arch[0])
