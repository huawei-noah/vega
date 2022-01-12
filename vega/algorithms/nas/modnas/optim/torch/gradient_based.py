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

"""Optimizer operating on tensor parameters."""

import math
import copy
from typing import Any, List, Optional, Tuple, Dict
import torch
from torch import Tensor
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer
from modnas.core.param_space import ParamSpace
from modnas.arch_space.mixed_ops import MixedOp
from modnas.registry.optim import register
from modnas.estim.base import EstimBase
from ..base import GradientBasedOptim


OPTIM_CONF_TYPE = Optional[Dict[str, Any]]


@register
class DARTSOptim(GradientBasedOptim):
    """Optimizer with DARTS algorithm.

    modified from https://github.com/khanrc/pt.darts
    """

    def __init__(
        self, a_optim: OPTIM_CONF_TYPE = None, w_momentum: float = 0.9, w_weight_decay: float = 0.0003,
        space: Optional[ParamSpace] = None
    ) -> None:
        super().__init__(space, a_optim)
        self.v_net = None
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def _virtual_step(self, trn_batch: Any, lr: float, optimizer: Optimizer, estim: EstimBase) -> None:
        # forward & calc loss
        model = estim.model
        loss = estim.loss(trn_batch, mode='train')  # L_trn(w)
        # compute gradient
        gradients = torch.autograd.grad(loss, model.parameters())
        # do virtual step (update gradient)
        with torch.no_grad():
            for w, vw, g in zip(model.parameters(), self.v_net.parameters(), gradients):
                m = optimizer.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - lr * (m + g + self.w_weight_decay * w))

    def step(self, estim: EstimBase) -> None:
        """Update Optimizer states using Estimator."""
        self.optim_reset()
        trn_batch = estim.get_cur_train_batch()
        val_batch = estim.get_next_valid_batch()
        lr = estim.trainer.get_lr()
        optimizer = estim.trainer.get_optimizer()
        model = estim.model
        if self.v_net is None:
            self.v_net = copy.deepcopy(model)
        # do virtual step (calc w`)
        self._virtual_step(trn_batch, lr, optimizer, estim)
        # calc unrolled loss
        loss = estim.loss(val_batch, model=self.v_net, mode='valid')  # L_val(w`)
        # compute gradient
        v_alphas = tuple(ParamSpace().tensor_values())
        v_weights = tuple(self.v_net.parameters())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = list(v_grads[len(v_alphas):])
        hessian = self._compute_hessian(dw, trn_batch, estim)
        # update final gradient = dalpha - lr*hessian
        with torch.no_grad():
            for a, da, h in zip(v_alphas, dalpha, hessian):
                a.grad = da - lr * h
        self.optim_step()

    def _compute_hessian(self, dw: List[Tensor], trn_batch: Tuple[Tensor, Tensor], estim: EstimBase) -> List[Any]:
        """Compute Hessian matrix.

        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        model = estim.model
        alphas = tuple(ParamSpace().tensor_values())
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = (0.01 / norm).item()
        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(model.parameters(), dw):
                p += eps * d
        loss = estim.loss(trn_batch, mode='train')
        dalpha_pos = torch.autograd.grad(loss, alphas)  # dalpha { L_trn(w+) }
        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(model.parameters(), dw):
                p -= 2. * eps * d
        loss = estim.loss(trn_batch, mode='train')
        dalpha_neg = torch.autograd.grad(loss, alphas)  # dalpha { L_trn(w-) }
        # recover w
        with torch.no_grad():
            for p, d in zip(model.parameters(), dw):
                p += eps * d
        hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian


@register
class BinaryGateOptim(GradientBasedOptim):
    """Optimizer with BinaryGate (ProxylessNAS) algorithm."""

    _default_optimizer_conf = {
        'type': 'Adam',
        'args': {
            'lr': 0.006,
            'betas': [0.0, 0.999],
            'weight_decay': 0,
        }
    }

    def __init__(
        self, a_optim: OPTIM_CONF_TYPE = None, n_samples: int = 2, renorm: bool = True,
        space: Optional[ParamSpace] = None
    ) -> None:
        super().__init__(space, a_optim or BinaryGateOptim._default_optimizer_conf)
        self.n_samples = n_samples
        self.sample = (self.n_samples != 0)
        self.renorm = renorm and self.sample

    def step(self, estim: EstimBase) -> None:
        """Update Optimizer states using Estimator."""
        self.optim_reset()
        model = estim.model
        val_batch = estim.get_next_valid_batch()
        # sample k
        if self.sample:
            for m in MixedOp.gen(model):
                m.sample_ops(n_samples=self.n_samples)
        # loss
        for m in MixedOp.gen(model):
            m.arch_param_grad(enabled=True)
        loss = estim.loss(val_batch, mode='valid')
        # backward
        loss.backward()
        # renormalization
        if not self.renorm:
            self.optim_step()
        else:
            with torch.no_grad():
                prev_pw = []
                for m in MixedOp.gen(model):
                    p = m.alpha()
                    s_op = m.s_op
                    pdt = p.detach()
                    pp = pdt.index_select(-1, torch.tensor(s_op).to(p.device))
                    if pp.size() == pdt.size():
                        continue
                    k = torch.sum(torch.exp(pdt)) / torch.sum(torch.exp(pp)) - 1
                    prev_pw.append(k)

            self.optim_step()

            with torch.no_grad():
                for kprev, m in zip(prev_pw, MixedOp.gen(model)):
                    p = m.alpha()
                    s_op = m.s_op
                    pdt = p.detach()
                    pp = pdt.index_select(-1, torch.tensor(s_op).to(p.device))
                    k = torch.sum(torch.exp(pdt)) / torch.sum(torch.exp(pp)) - 1
                    for i in s_op:
                        p[i] += (torch.log(k) - torch.log(kprev))

        for m in MixedOp.gen(model):
            m.arch_param_grad(enabled=False)
            m.reset_ops()


@register
class DirectGradOptim(GradientBasedOptim):
    """Optimizer by backwarding training loss."""

    def __init__(self, a_optim: OPTIM_CONF_TYPE = None, space: Optional[ParamSpace] = None) -> None:
        super().__init__(space, a_optim)

    def step(self, estim: EstimBase) -> None:
        """Update Optimizer states using Estimator."""
        self.optim_step()
        self.optim_reset()


@register
class DirectGradBiLevelOptim(GradientBasedOptim):
    """Optimizer by backwarding validating loss."""

    def __init__(self, a_optim: OPTIM_CONF_TYPE = None, space: Optional[ParamSpace] = None) -> None:
        super().__init__(space, a_optim)

    def step(self, estim: EstimBase) -> None:
        """Update Optimizer states using Estimator."""
        self.optim_reset()
        loss = estim.loss(estim.get_next_valid_batch(), mode='valid')
        loss.backward()
        self.optim_step()


@register
class REINFORCEOptim(GradientBasedOptim):
    """Optimizer with REINFORCE algorithm.

    modified from https://github.com/mit-han-lab/proxylessnas
    """

    def __init__(
        self, a_optim: OPTIM_CONF_TYPE = None, batch_size: int = 10,
        space: Optional[ParamSpace] = None
    ) -> None:
        super().__init__(space, a_optim)
        self.batch_size = batch_size
        self.baseline = None
        self.baseline_decay_weight = 0.99

    def step(self, estim: EstimBase) -> None:
        """Update Optimizer states using Estimator."""
        model = estim.model
        self.optim_reset()
        grad_batch = []
        reward_batch = []
        for _ in range(self.batch_size):
            # calculate reward according to net_info
            reward = estim.get_score(estim.compute_metrics())
            # loss term
            obj_term = 0
            for m in MixedOp.gen(model):
                p = m.alpha()
                if p.grad is not None:
                    p.grad.data.zero_()
                path_prob = m.prob()
                smpl = m.s_path_f
                path_prob_f = path_prob.index_select(-1, torch.tensor(smpl).to(path_prob.device))
                obj_term = obj_term + torch.log(path_prob_f)
            loss = -obj_term
            # backward
            loss.backward()
            # take out gradient dict
            grad_list = []
            for m in MixedOp.gen(model):
                p = m.alpha()
                grad_list.append(p.grad.data.clone())
            grad_batch.append(grad_list)
            reward_batch.append(reward)

        # update baseline function
        avg_reward = sum(reward_batch) / self.batch_size
        if self.baseline is None:
            self.baseline = avg_reward
        else:
            self.baseline += self.baseline_decay_weight * (avg_reward - self.baseline)
        # assign gradients
        for idx, m in enumerate(MixedOp.gen(model)):
            p = m.alpha()
            p.grad.data.zero_()
            for j in range(self.batch_size):
                p.grad.data += (reward_batch[j] - self.baseline) * grad_batch[j][idx]
            p.grad.data /= self.batch_size
        # apply gradients
        self.optim_step()


@register
class GumbelAnnealingOptim(GradientBasedOptim):
    """Optimizer with Gumbel Annealing (SNAS) algorithm."""

    def __init__(self,
                 a_optim: OPTIM_CONF_TYPE = None,
                 init_temp: float = 1e4,
                 exp_anneal_rate: float = 0.0015,
                 anneal_interval: int = 1,
                 restart_period: Optional[int] = None,
                 space: Optional[ParamSpace] = None) -> None:
        super().__init__(space, a_optim)
        self.init_temp = init_temp
        self.exp_anneal_rate = exp_anneal_rate
        self.temp = self.init_temp
        self.restart_period = restart_period or 0
        self.anneal_interval = anneal_interval
        self.cur_step = 0

    def step(self, estim: EstimBase) -> None:
        """Update Optimizer states using Estimator."""
        self.optim_reset()
        model = estim.model
        self._apply_temp(model)
        loss = estim.loss(estim.get_next_valid_batch(), mode='valid')
        loss.backward()
        self.optim_step()
        self.cur_step += 1
        if self.restart_period > 0 and self.cur_step >= self.restart_period:
            self.cur_step = 0
        intv = self.anneal_interval
        if self.cur_step % intv == 0:
            self.temp = self.init_temp * math.exp(-self.exp_anneal_rate * self.cur_step / intv)

    def _apply_temp(self, model: Module) -> None:
        for m in MixedOp.gen(model):
            m.set_temperature(self.temp)
