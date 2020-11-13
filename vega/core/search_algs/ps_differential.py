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
from functools import partial
import vega
from .search_algorithm import SearchAlgorithm
from zeus.common import ClassFactory, ClassType
from zeus.networks.network_desc import NetworkDesc
from zeus.trainer.conf import TrainerConfig
from zeus.common import ConfigSerializable

if vega.is_torch_backend():
    import torch
elif vega.is_tf_backend():
    import tensorflow as tf


def _concat(xs):
    """Concat Tensor."""
    return torch.cat([x.view(-1) for x in xs])


class DifferentialConfig(ConfigSerializable):
    """Config for Differential."""

    sample_num = 1
    momentum = 0.9
    weight_decay = 3.0e-4
    parallel = False
    codec = 'DartsCodec'
    arch_optim = dict(type='Adam', lr=3.0e-4, betas=[0.5, 0.999], weight_decay=1.0e-3)
    criterion = dict(type='CrossEntropyLoss')
    tf_arch_optim = dict(type='AdamOptimizer', learning_rate=3.0e-4, beta1=0.5, beta2=0.999)
    tf_criterion = dict(type='CrossEntropyLoss')
    objective_keys = 'accuracy'


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class DifferentialAlgorithm(SearchAlgorithm):
    """Differential algorithm.

    :param search_space: Input search_space.
    :type search_space: SearchSpace
    """

    config = DifferentialConfig()
    trainer_config = TrainerConfig()

    def __init__(self, search_space=None):
        """Init DifferentialAlgorithm."""
        super(DifferentialAlgorithm, self).__init__(search_space)
        self.network_momentum = self.config.momentum
        self.network_weight_decay = self.config.weight_decay
        self.parallel = self.config.parallel
        self.criterion = self.config.criterion
        self.sample_num = self.config.sample_num
        self.sample_idx = 0

    def set_model(self, model):
        """Set model."""
        self.model = model
        if vega.is_torch_backend():
            if self.parallel:
                self.model = self.model.module
            self.loss = self._init_loss().cuda()
            self.optimizer = self._init_arch_optimizer(self.model)

    def _init_arch_optimizer(self, model=None):
        """Init arch optimizer."""
        if vega.is_torch_backend():
            optim_config = self.config.arch_optim.copy()
            optim_name = optim_config.pop('type')
            optim_class = getattr(importlib.import_module('torch.optim'), optim_name)
            learnable_params = [model.alphas_normal, model.alphas_reduce]
            optimizer = optim_class(learnable_params, **optim_config)
        elif vega.is_tf_backend():
            optim_config = self.config.tf_arch_optim.copy()
            optim_name = optim_config.pop('type')
            optim_config['learning_rate'] = self.lr
            optim_class = getattr(importlib.import_module('tensorflow.compat.v1.train'),
                                  optim_name)
            optimizer = optim_class(**optim_config)
        return optimizer

    def _init_loss(self):
        """Init loss."""
        if vega.is_torch_backend():
            loss_config = self.criterion.copy()
            loss_name = loss_config.pop('type')
            loss_class = getattr(importlib.import_module('torch.nn'), loss_name)
            return loss_class(**loss_config)
        elif vega.is_tf_backend():
            from inspect import isclass
            loss_config = self.config.tf_criterion.copy()
            loss_name = loss_config.pop('type')
            if ClassFactory.is_exists('trainer.loss', loss_name):
                loss_class = ClassFactory.get_cls('trainer.loss', loss_name)
                if isclass(loss_class):
                    return loss_class(**loss_config)
                else:
                    return partial(loss_class, **loss_config)
            else:
                loss_class = getattr(importlib.import_module('tensorflow.losses'), loss_name)
                return partial(loss_class, **loss_config)

    def step(self, train_x=None, train_y=None, valid_x=None, valid_y=None,
             lr=None, w_optimizer=None, w_loss=None, unrolled=None, scope_name=None):
        """Compute one step."""
        if vega.is_torch_backend():
            self.optimizer.zero_grad()
            loss = w_loss(self.model(valid_x), valid_y)
            loss.backward()
            self.optimizer.step()
            return
        elif vega.is_tf_backend():
            self.lr = lr
            global_step = tf.compat.v1.train.get_global_step()
            loss_fn = self._init_loss()
            self.optimizer = self._init_arch_optimizer()
            logits = self.model(valid_x)
            logits = tf.cast(logits, tf.float32)
            loss = loss_fn(logits, valid_y)
            loss_scale = self.trainer_config.loss_scale if self.trainer_config.amp else 1.
            arch_op = self.model.get_weight_ops()
            if loss_scale != 1:
                scaled_grad_vars = self.optimizer.compute_gradients(loss * loss_scale, var_list=arch_op)
                unscaled_grad_vars = [(grad / loss_scale, var) for grad, var in scaled_grad_vars]
                minimize_op = self.optimizer.apply_gradients(unscaled_grad_vars, global_step)
            else:
                grad_vars = self.optimizer.compute_gradients(loss, var_list=arch_op)
                minimize_op = self.optimizer.apply_gradients(grad_vars, global_step)
            return minimize_op

    def _backward_step(self, input_valid, target_valid):
        """Compute backward_step."""
        loss = self.loss(self.model(input_valid), target_valid)
        loss.backward()

    def new_model(self):
        """Build new model."""
        net_desc = NetworkDesc(self.search_space)
        model_new = net_desc.to_model().cuda()
        for x, y in zip(model_new.arch_parameters(), self.model.arch_parameters()):
            x.detach().copy_(y.detach())
        return model_new

    def search(self):
        """Search function."""
        return self.sample_idx, self.search_space

    @property
    def is_completed(self):
        """Check if the search is finished."""
        self.sample_idx += 1
        return self.sample_idx > self.sample_num
