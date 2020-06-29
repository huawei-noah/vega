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
from .search_algorithm import SearchAlgorithm
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.networks import NetworkDesc
from vega.search_space.codec import Codec
from vega.core.common.file_ops import FileOps


def _concat(xs):
    """Concat Tensor."""
    return torch.cat([x.view(-1) for x in xs])


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class DifferentialAlgorithm(SearchAlgorithm):
    """Differential algorithm.

    :param search_space: Input search_space.
    :type search_space: SearchSpace
    """

    def __init__(self, search_space=None):
        """Init DifferentialAlgorithm."""
        super(DifferentialAlgorithm, self).__init__(search_space)
        self.args = self.cfg
        self.ss = search_space
        self.network_momentum = self.args.momentum
        self.network_weight_decay = self.args.weight_decay
        self.parallel = self.args.parallel
        self.codec = Codec(self.args.codec, self.ss)
        self.sample_num = self.args.get('sample_num', 1)
        self.sample_idx = 0

    def set_model(self, model):
        """Set model."""
        self.model = model
        if self.args.parallel:
            self.module = self.model.module
        else:
            self.module = self.model
        self.loss = self._init_loss().cuda()
        self.optimizer = self._init_arch_optimizer(self.module)

    def _init_arch_optimizer(self, model):
        """Init arch optimizer."""
        optim_config = self.args.arch_optim.copy()
        optim_name = optim_config.pop('type')
        optim_class = getattr(
            importlib.import_module('torch.optim'), optim_name)
        learnable_params = model.arch_parameters()
        return optim_class(learnable_params, **optim_config)

    def _init_loss(self):
        """Init loss."""
        loss_config = self.args.criterion.copy()
        loss_name = loss_config.pop('type')
        loss_class = getattr(importlib.import_module('torch.nn'), loss_name)
        return loss_class(**loss_config)

    def step(self, train_x, train_y, valid_x, valid_y,
             lr, w_optimizer, w_loss, unrolled):
        """Compute one step."""
        self.optimizer.zero_grad()
        loss = w_loss(self.module(valid_x), valid_y)
        loss.backward()
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        """Compute backward_step."""
        loss = self.loss(self.module(input_valid), target_valid)
        loss.backward()

    def new_model(self):
        """Build new model."""
        net_desc = NetworkDesc(self.ss.search_space)
        model_new = net_desc.to_model().cuda()
        for x, y in zip(model_new.arch_parameters(), self.module.arch_parameters()):
            x.detach().copy_(y.detach())
        return model_new

    def search(self):
        """Search function."""
        return self.sample_idx, NetworkDesc(self.ss.search_space)

    def update(self, worker_path):
        """Update function.

        :param worker_path: the worker_path that saved `performance.txt`.
        :type worker_path: str
        """
        if self.backup_base_path is not None:
            FileOps.copy_folder(self.local_base_path, self.backup_base_path)

    @property
    def is_completed(self):
        """Check if the search is finished."""
        self.sample_idx += 1
        return self.sample_idx > self.sample_num
