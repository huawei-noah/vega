# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""DifferentialAlgorithm."""
import importlib
import torch
import numpy as np
from vega.search_space.search_algs.search_algorithm import SearchAlgorithm
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.networks import NetworkDesc
from vega.search_space.codec import Codec
import logging


def _concat(xs):
    """Concat Tensor."""
    return torch.cat([x.view(-1) for x in xs])


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class CARSAlgorithm(SearchAlgorithm):
    """Differential algorithm.

    :param search_space: Input search_space.
    :type search_space: SearchSpace
    """

    def __init__(self, search_space=None):
        """Init CARSAlgorithm."""
        super(CARSAlgorithm, self).__init__(search_space)
        self.ss = search_space
        self.network_momentum = self.policy.momentum
        self.network_weight_decay = self.policy.weight_decay
        self.parallel = self.policy.parallel
        self.codec = Codec(self.cfg.codec, self.ss)
        self.sample_num = self.policy.get('sample_num', 1)
        self.sample_idx = 0
        self.completed = False

    def set_model(self, model):
        """Set model."""
        self.model = model
        if self.policy.parallel:
            self.module = self.model.module
        else:
            self.module = self.model
        self.criterion = self._init_loss().cuda()
        self.optimizer = self._init_arch_optimizer(self.module)

    def _init_arch_optimizer(self, model):
        """Init arch optimizer."""
        optim_config = self.policy.arch_optim.copy()
        optim_name = optim_config.pop('type')
        optim_class = getattr(
            importlib.import_module('torch.optim'), optim_name)
        learnable_params = model.arch_parameters()
        return optim_class(learnable_params, **optim_config)

    def _init_loss(self):
        """Init loss."""
        loss_config = self.policy.criterion.copy()
        loss_name = loss_config.pop('type')
        loss_class = getattr(importlib.import_module('torch.nn'), loss_name)
        return loss_class(**loss_config)

    def new_model(self):
        """Build new model."""
        net_desc = NetworkDesc(self.ss.search_space)
        model_new = net_desc.to_model().cuda()
        for x, y in zip(model_new.arch_parameters(), self.module.arch_parameters()):
            x.detach().copy_(y.detach())
        return model_new

    def search(self):
        """Search function."""
        logging.info('====> {}.search()'.format(__name__))
        self.completed = True
        return self.sample_idx, NetworkDesc(self.ss.search_space)

    def update(self, worker_path):
        """Update function.

        :param worker_path: the worker_path that saved `performance.txt`.
        :type worker_path: str
        """
        self.sample_idx += 1

    def gen_offspring(self, alphas, offspring_ratio=1.0):
        """Generate offsprings.

        :param alphas: Parameteres for populations
        :type alphas: nn.Tensor
        :param offspring_ratio: Expanding ratio
        :type offspring_ratio: float
        :return: The generated offsprings
        :rtype: nn.Tensor
        """
        n_offspring = int(offspring_ratio * alphas.size(0))
        offsprings = []
        while len(offsprings) != n_offspring:
            rand = np.random.rand()
            if rand < 0.25:
                alphas_c = self.model.mutation(alphas[np.random.randint(0, alphas.size(0))])
            elif rand < 0.5:
                alphas_c = self.model.crossover(
                    alphas[np.random.randint(0, alphas.size(0))],
                    alphas[np.random.randint(0, alphas.size(0))])
            else:
                alphas_c = self.model.random_single_path().to(alphas.device)
            if self.judge_repeat(alphas, alphas_c) == 0:
                offsprings.append(alphas_c)
        offsprings = torch.cat([offspring.unsqueeze(0) for offspring in offsprings], dim=0)
        return offsprings

    def judge_repeat(self, alphas, new_alphas):
        """Judge if two individuals are the same.

        :param alphas_a: An individual
        :type alphas_a: nn.Tensor
        :param new_alphas: An individual
        :type new_alphas: nn.Tensor
        :return: True or false
        :rtype: nn.Tensor
        """
        diff = (alphas - new_alphas.unsqueeze(0)).abs().view(alphas.size(0), -1)
        diff = diff.sum(1)
        return (diff == 0).sum()

    @property
    def is_completed(self):
        """Check if the search is finished."""
        return (self.sample_idx >= self.sample_num) or self.completed
