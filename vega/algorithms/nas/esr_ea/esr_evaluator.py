# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""EsrGpuEvaluator used to do evaluate process on gpu."""
import os
import time
import math
import logging
import errno
import pickle
import torch
import numpy as np
from copy import deepcopy
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.trainer.pytorch import Trainer
from vega.core.trainer.utils import WorkerTypes
from vega.core.common import FileOps, Config
from vega.search_space.codec import Codec
from vega.search_space.networks import NetworkDesc
from vega.core.evaluator.gpu_evaluator import GpuEvaluator
from .esr_ea_individual import ESRIndividual
from vega.core.metrics.pytorch import Metrics


logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.GPU_EVALUATOR)
class EsrGpuEvaluator(GpuEvaluator):
    """Evaluator is a gpu evaluator.

    :param args: arguments from user and default config file
    :type args: dict or Config, default to None
    :param train_data: training dataset
    :type train_data: torch dataset, default to None
    :param valid_data: validate dataset
    :type valid_data: torch dataset, default to None
    :param worker_info: the dict worker info of workers that finished train.
    :type worker_info: dict or None.

    """

    def __init__(self, worker_info=None, model=None, hps=None, **kwargs):
        """Init GpuEvaluator."""
        super(EsrGpuEvaluator, self).__init__(self.cfg)

    def _init_model(self):
        """Initialize the model architecture for full train step.

        :return: train model
        :rtype: class
        """
        model_cfg = ClassFactory.__configs__.get('model')
        if 'model_desc' in model_cfg and model_cfg.model_desc is not None:
            model_desc = model_cfg.model_desc
        else:
            raise ValueError('Model_desc is None for evaluator')
        search_space = Config({"search_space": model_desc})
        self.codec = Codec(self.cfg.codec, search_space)
        self._get_selected_arch()
        indiv_cfg = self.codec.decode(self.elitism)
        logger.info('Model arch:{}'.format(self.elitism.active_net_list()))
        self.model_desc = self.elitism.active_net_list()
        net_desc = NetworkDesc(indiv_cfg)
        model = net_desc.to_model()
        return model

    def _get_selected_arch(self):
        self.elitism = ESRIndividual(self.codec, deepcopy(self.cfg))
        if "model_arch" in self.cfg and self.cfg.model_arch is not None:
            self.elitism.update_gene(self.cfg.model_arch)
        else:
            sel_arch_file = self.cfg.model_desc_file
            sel_arch = np.load(sel_arch_file)
            self.elitism.update_gene(sel_arch[0])

    def valid(self, loader):
        """Validate one step of model.

        :param loader: validation dataloader
        """
        metrics = Metrics(self.cfg.metric)
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                img_lr, img_hr = batch["LR"].cuda(), batch["HR"].cuda()
                image_sr = self.model(img_lr)
                metrics(image_sr, img_hr)
        performance = metrics.results
        logging.info('Valid metric: {}'.format(performance))
        return performance
