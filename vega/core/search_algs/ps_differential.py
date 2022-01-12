# -*- coding:utf-8 -*-

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

"""DifferentialAlgorithm."""
import importlib
import math
import logging
from functools import partial
import numpy as np
import vega
from vega.common import ClassFactory, ClassType
from vega.networks.network_desc import NetworkDesc
from vega.trainer.conf import TrainerConfig
from vega.common import ConfigSerializable
from .search_algorithm import SearchAlgorithm

if vega.is_torch_backend():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.distributions.categorical as cate
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
    # config for sgas algorithm
    use_history = True
    history_size = 5
    warmup_dec_epoch = 9
    decision_freq = 5


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
        # sgas config
        self.use_history = self.config.use_history
        self.history_size = self.config.history_size
        self.warmup_dec_epoch = self.config.warmup_dec_epoch
        self.decision_freq = self.config.decision_freq

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
            learnable_params = getattr(self.model, 'learnable_params',
                                       [model.alphas_normal, model.alphas_reduce])
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
        def set_opt_requires_grad(value):
            for param in self.optimizer.param_groups:
                for parameter in param['params']:
                    parameter.requires_grad = value
        if vega.is_torch_backend():
            set_opt_requires_grad(True)
            self.optimizer.zero_grad()
            loss = w_loss(self.model(valid_x), valid_y)
            loss.backward()
            self.optimizer.step()
            set_opt_requires_grad(False)
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

    def edge_decision(self, type, alphas, selected_idxs, candidate_flags, probs_history, epoch):
        """Calculate the decision for each edge.

        :param type: the type of cell
        :type type: str ('normal' or 'reduce')
        """
        mat = F.softmax(torch.stack(alphas, dim=0), dim=-1).detach()
        logging.info('alpha: {}'.format(mat))
        importance = torch.sum(mat[:, 1:], dim=-1)
        logging.info(type + " importance {}".format(importance))

        probs = mat[:, 1:] / importance[:, None]
        logging.info(type + " probs {}".format(probs))
        entropy = cate.Categorical(probs=probs).entropy() / math.log(probs.shape[1])
        logging.info(type + " entropy {}".format(entropy))

        if self.use_history:
            # SGAS Cri.2
            logging.info(type + " probs history {}".format(probs_history))
            histogram_inter = self.histogram_average(probs_history, probs)
            logging.info(type + " histogram intersection average {}".format(histogram_inter))
            probs_history.append(probs)
            if (len(probs_history) > self.history_size):
                probs_history.pop(0)

            score = self.normalize(importance) * self.normalize(1 - entropy) * self.normalize(histogram_inter)
            logging.info(type + " score {}".format(score))
        else:
            # SGAS Cri.1
            score = self.normalize(importance) * self.normalize(1 - entropy)
            logging.info(type + " score {}".format(score))

        if torch.sum(candidate_flags.int()) > 0 and epoch >= self.warmup_dec_epoch and \
                (epoch - self.warmup_dec_epoch) % self.decision_freq == 0:
            masked_score = torch.min(score, (2 * candidate_flags.float() - 1) * np.inf)
            selected_edge_idx = torch.argmax(masked_score)
            # add 1 since none op
            selected_op_idx = torch.argmax(probs[selected_edge_idx]) + 1
            selected_idxs[selected_edge_idx] = selected_op_idx

            candidate_flags[selected_edge_idx] = False
            alphas[selected_edge_idx].requires_grad = False
            if type == 'normal':
                reduction = False
            elif type == 'reduce':
                reduction = True
            else:
                raise Exception('Unknown Cell Type')
            candidate_flags, selected_idxs = self.check_edges(candidate_flags, selected_idxs, reduction=reduction)
            logging.info("#" * 30 + " Decision Epoch " + "#" * 30)
            logging.info("epoch {}, {}_selected_idxs {}, added edge {} with op idx {}".format(
                epoch, type, selected_idxs, selected_edge_idx, selected_op_idx))
            logging.info(type + "_candidate_flags {}".format(candidate_flags))
            return True, selected_idxs, candidate_flags

        else:
            logging.info("#" * 30 + " Not a Decision Epoch " + "#" * 30)
            logging.info("epoch {}, {}_selected_idxs {}".format(epoch, type, selected_idxs))
            logging.info(type + "_candidate_flags {}".format(candidate_flags))
            return False, selected_idxs, candidate_flags

    def normalize(self, alphas, epsilon=1e-9):
        """Normalize alphas."""
        min_ = torch.min(alphas, dim=-1, keepdim=True)[0] + epsilon
        max_ = torch.max(alphas, dim=-1, keepdim=True)[0] + epsilon
        range_ = max_ - min_ + epsilon
        return (alphas - min_) / range_

    def histogram_average(self, history, probs):
        """Calculate the average history information using the histogram intersection method."""
        def histogram_intersection(a, b):
            c = np.minimum(a.detach().cpu().numpy(), b.detach().cpu().numpy())
            c = torch.from_numpy(c).cuda()
            sums = c.sum(dim=1)
            return sums
        histogram_inter = torch.zeros(probs.shape[0], dtype=torch.float).cuda()
        if not history:
            return histogram_inter
        for hist in history:
            histogram_inter += histogram_intersection(hist, probs)
        histogram_inter /= len(history)
        return histogram_inter

    def check_edges(self, flags, selected_idxs, reduction=False):
        """Check and prune edge."""
        n = 2
        max_num_edges = 2
        start = 0
        for i in range(self.model.steps):
            end = start + n
            num_selected_edges = torch.sum(1 - flags[start:end].int())
            if num_selected_edges >= max_num_edges:
                for j in range(start, end):
                    if flags[j]:
                        flags[j] = False
                        # pruned edges PRIMITIVES.index('none')
                        selected_idxs[j] = 0
                        if reduction:
                            self.model.alphas_reduce[j].requires_grad = False
                        else:
                            self.model.alphas_normal[j].requires_grad = False
                    else:
                        pass
            start = end
            n += 1

        return flags, selected_idxs
