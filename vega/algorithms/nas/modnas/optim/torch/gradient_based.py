# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Optimizer operating on tensor parameters."""
import math
import copy
import torch
from ..base import GradientBasedOptim
from modnas.core.param_space import ParamSpace
from modnas.arch_space.mixed_ops import MixedOp
from modnas.registry.optim import register


@register
class DARTSOptim(GradientBasedOptim):
    """Optimizer with DARTS algorithm.

    modified from https://github.com/khanrc/pt.darts
    """

    def __init__(self, a_optim=None, w_momentum=0.9, w_weight_decay=0.0003, space=None):
        super().__init__(space, a_optim)
        self.v_net = None
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def _virtual_step(self, trn_batch, lr, optimizer, estim):
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

    def step(self, estim):
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
        alphas = ParamSpace().tensor_values()
        v_alphas = tuple(alphas)
        v_weights = tuple(self.v_net.parameters())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]
        hessian = self._compute_hessian(dw, trn_batch, estim)
        # update final gradient = dalpha - lr*hessian
        with torch.no_grad():
            for a, da, h in zip(alphas, dalpha, hessian):
                a.grad = da - lr * h
        self.optim_step()

    def _compute_hessian(self, dw, trn_batch, estim):
        """Compute Hessian matrix.

        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        model = estim.model
        alphas = ParamSpace().tensor_values()
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
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
        hessian = [(p - n) / 2. * eps.item() for p, n in zip(dalpha_pos, dalpha_neg)]
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

    def __init__(self, a_optim=None, n_samples=2, renorm=True, space=None):
        super().__init__(space, a_optim or BinaryGateOptim._default_optimizer_conf)
        self.n_samples = n_samples
        self.sample = (self.n_samples != 0)
        self.renorm = renorm and self.sample

    def step(self, estim):
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

    def __init__(self, a_optim=None, space=None):
        super().__init__(space, a_optim)

    def step(self, estim):
        """Update Optimizer states using Estimator."""
        self.optim_step()
        self.optim_reset()


@register
class DirectGradBiLevelOptim(GradientBasedOptim):
    """Optimizer by backwarding validating loss."""

    def __init__(self, a_optim=None, space=None):
        super().__init__(space, a_optim)

    def step(self, estim):
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

    def __init__(self, a_optim=None, batch_size=10, space=None):
        super().__init__(space, a_optim)
        self.batch_size = batch_size
        self.baseline = None
        self.baseline_decay_weight = 0.99

    def step(self, estim):
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
                 a_optim=None,
                 init_temp=1e4,
                 exp_anneal_rate=0.0015,
                 anneal_interval=1,
                 restart_period=None,
                 space=None):
        super().__init__(space, a_optim)
        self.init_temp = init_temp
        self.exp_anneal_rate = exp_anneal_rate
        self.temp = self.init_temp
        if restart_period is None:
            restart_period = 0
        self.restart_period = int(restart_period)
        self.anneal_interval = anneal_interval
        self.cur_step = 0

    def step(self, estim):
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

    def _apply_temp(self, model):
        for m in MixedOp.gen(model):
            m.set_temperature(self.temp)
