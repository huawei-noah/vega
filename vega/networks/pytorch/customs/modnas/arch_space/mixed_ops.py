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

"""Mixed operators."""
from collections import OrderedDict
from typing import Any, Collection, Iterator, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.module import Module
from modnas.core.params.base import Param
from modnas.core.params import Categorical
from modnas.registry.params import build
from modnas.registry.arch_space import register
from modnas.utils.logging import get_logger


logger = get_logger('arch_space')


class MixedOp(nn.Module):
    """Base Mixed operator class."""

    def __init__(
        self, candidates: Union[OrderedDict, Collection[Tuple[str, Module]]], arch_param: Optional[Param]
    ) -> None:
        super().__init__()
        if isinstance(candidates, (tuple, list)):
            candidates = {n: p for n, p in candidates}
        if isinstance(candidates, dict):
            self._ops = nn.ModuleDict(candidates)
        else:
            raise ValueError('unsupported candidates type')
        if arch_param is None:
            arch_param = build('TorchTensor', len(self._ops))
        self.arch_param = arch_param
        logger.debug('mixed op: {} p: {}'.format(type(self), arch_param))

    def candidates(self) -> Any:
        """Return list of candidate operators."""
        return list(self._ops.values())

    def candidate_names(self) -> List[str]:
        """Return list of candidate operator names."""
        return list(self._ops.keys())

    def named_candidates(self):
        """Return an iterator over named candidate operators."""
        for n, cand in self._ops.items():
            yield n, cand

    def alpha(self) -> Any:
        """Return architecture parameter value."""
        return self.arch_param_value()

    def prob(self) -> Tensor:
        """Return candidate probabilities."""
        return F.softmax(self.alpha(), dim=-1)

    def arch_param_value(self) -> Any:
        """Return architecture parameter value."""
        return self.arch_param.value()

    def to_arch_desc(self, *args, **kwargs):
        """Return archdesc from mixed operator."""
        raise NotImplementedError

    @staticmethod
    def gen(model: Module) -> Iterator[Module]:
        """Return an iterator over all MixedOp in a model."""
        for m in model.modules():
            if isinstance(m, MixedOp):
                yield m


@register
class SoftmaxSumMixedOp(MixedOp):
    """Mixed operator using softmax weighted sum."""

    def __init__(self, candidates: OrderedDict, arch_param: Optional[Param] = None) -> None:
        super().__init__(candidates, arch_param)

    def forward(self, *args, **kwargs) -> Union[Tensor, int]:
        """Compute MixedOp output."""
        outputs = [op(*args, **kwargs) for op in self.candidates()]
        w_path = F.softmax(self.alpha().to(device=outputs[0].device), dim=-1)
        return sum((w * o for w, o in zip(w_path, outputs)))

    def to_arch_desc(self, k: int = 1) -> Any:
        """Return archdesc from mixed operator."""
        cname = self.candidate_names()
        w = F.softmax(self.alpha().detach(), dim=-1)
        _, cand_idx = torch.topk(w, k)
        desc = [cname[i] for i in cand_idx]
        if desc == []:
            return [None]
        return desc


@register
class BinaryGateMixedOp(MixedOp):
    """Mixed operator controlled by BinaryGate."""

    def __init__(self, candidates: OrderedDict, arch_param: Optional[Param] = None, n_samples: int = 1) -> None:
        super().__init__(candidates, arch_param)
        self.n_samples = n_samples
        self.s_path_f = []
        self.last_samples = []
        self.s_op = []
        self.a_grad_enabled = False
        self.reset_ops()

    def arch_param_grad(self, enabled: bool) -> None:
        """Set if enable architecture parameter grad."""
        self.a_grad_enabled = enabled

    def sample_path(self) -> None:
        """Sample candidates in forward pass."""
        p = self.alpha()
        s_op = self.s_op
        self.w_path_f = F.softmax(p.index_select(-1, torch.tensor(s_op).to(p.device)), dim=-1)
        samples = self.w_path_f.multinomial(1 if self.a_grad_enabled else self.n_samples)
        self.s_path_f = [s_op[i] for i in samples]

    def sample_ops(self, n_samples: int) -> None:
        """Sample activated candidates."""
        samples = self.prob().multinomial(n_samples).detach()
        self.s_op = list(samples.flatten().cpu().numpy())

    def reset_ops(self) -> None:
        """Reset activated candidates."""
        s_op = list(range(len(self.candidates())))
        self.last_samples = s_op
        self.s_op = s_op

    def forward(self, *args, **kwargs) -> Tensor:
        """Compute MixedOp output."""
        self.sample_path()
        s_path_f = self.s_path_f
        candidates = self.candidates()
        if self.training:
            self.swap_ops(s_path_f)
        if self.a_grad_enabled:
            p = self.alpha()
            ctx_dict = {
                's_op': self.s_op,
                's_path_f': self.s_path_f,
                'w_path_f': self.w_path_f,
                'candidates': candidates,
            }
            m_out = BinaryGateFunction.apply(kwargs, p, ctx_dict, *args)
        else:
            outputs = [candidates[i](*args, **kwargs) for i in s_path_f]
            m_out = sum(outputs) if len(s_path_f) > 1 else outputs[0]
        self.last_samples = s_path_f
        return m_out

    def swap_ops(self, samples: List[int]) -> None:
        """Remove unused candidates from computation graph."""
        cands = self.candidates()
        for i in self.last_samples:
            if i in samples:
                continue
            for p in cands[i].parameters():
                if p.grad_fn is not None:
                    continue
                p.requires_grad = False
                p.grad = None
        for i in samples:
            for p in cands[i].parameters():
                if p.grad_fn is not None:
                    continue
                p.requires_grad_(True)

    def to_arch_desc(self, k: int = 1) -> Any:
        """Return archdesc from mixed operator."""
        cname = self.candidate_names()
        w = F.softmax(self.alpha().detach(), dim=-1)
        _, cand_idx = torch.topk(w, k)
        desc = [cname[i] for i in cand_idx]
        if desc == []:
            return [None]
        return desc


class BinaryGateFunction(torch.autograd.function.Function):
    """BinaryGate gradient approximation function."""

    @staticmethod
    def forward(ctx, kwargs, alpha, ctx_dict, *args):
        """Return forward outputs."""
        ctx.__dict__.update(ctx_dict)
        ctx.kwargs = kwargs
        ctx.param_shape = alpha.shape
        candidates = ctx.candidates
        s_path_f = ctx.s_path_f
        with torch.enable_grad():
            if len(s_path_f) == 1:
                m_out = candidates[s_path_f[0]](*args, **kwargs)
            else:
                m_out = sum(candidates[i](*args, **kwargs) for i in s_path_f)
        ctx.save_for_backward(*args, m_out)
        return m_out.data

    @staticmethod
    def backward(ctx, m_grad):
        """Return backward outputs."""
        args_f = ctx.saved_tensors[:-1]
        m_out = ctx.saved_tensors[-1]
        retain = True if len(args_f) > 1 else False
        grad_args = torch.autograd.grad(m_out, args_f, m_grad, only_inputs=True, retain_graph=retain)
        with torch.no_grad():
            a_grad = torch.zeros(ctx.param_shape)
            s_op = ctx.s_op
            w_path_f = ctx.w_path_f
            s_path_f = ctx.s_path_f
            kwargs = ctx.kwargs
            candidates = ctx.candidates
            for j, oj in enumerate(s_op):
                if oj in s_path_f:
                    op_out = m_out.data
                else:
                    op = candidates[oj]
                    op_out = op(*args_f, **kwargs)
                g_grad = torch.sum(m_grad * op_out)
                for i, oi in enumerate(s_op):
                    kron = 1 if i == j else 0
                    a_grad[oi] = a_grad[oi] + g_grad * w_path_f[j] * (kron - w_path_f[i])
        return (None, a_grad, None) + grad_args


@register
class BinaryGateUniformMixedOp(BinaryGateMixedOp):
    """Mixed operator controlled by BinaryGate, which candidates sampled uniformly."""

    def sample_path(self) -> None:
        """Sample candidates in forward pass."""
        p = self.alpha()
        s_op = self.s_op
        self.w_path_f = F.softmax(p.index_select(-1, torch.tensor(s_op).to(p.device)), dim=-1)
        samples = F.softmax(torch.ones(len(s_op)), dim=-1).multinomial(self.n_samples)
        s_path_f = [s_op[i] for i in samples]
        self.s_path_f = s_path_f

    def sample_ops(self, n_samples: int) -> None:
        """Sample activated candidates."""
        p = self.alpha()
        samples = F.softmax(torch.ones(p.shape), dim=-1).multinomial(n_samples).detach()
        self.s_op = list(samples.flatten().cpu().numpy())


@register
class GumbelSumMixedOp(MixedOp):
    """Mixed operator using gumbel softmax sum."""

    def __init__(self, candidates: OrderedDict, arch_param: Optional[Param] = None) -> None:
        super().__init__(candidates, arch_param)
        self.temp = 1e5

    def set_temperature(self, temp: float) -> None:
        """Set annealing temperature."""
        self.temp = temp

    def prob(self) -> Tensor:
        """Return candidate probabilities."""
        p = self.alpha()
        eps = 1e-7
        uniforms = torch.rand(p.shape, device=p.device).clamp(eps, 1 - eps)
        gumbels = -((-(uniforms.log())).log())
        scores = (p + gumbels) / self.temp
        return F.softmax(scores, dim=-1)

    def forward(self, *args, **kwargs) -> Union[Tensor, int]:
        """Compute MixedOp output."""
        outputs = [op(*args, **kwargs) for op in self.candidates()]
        w_path = self.prob().to(outputs[0].device)
        return sum(w * o for w, o in zip(w_path, outputs))

    def to_arch_desc(self, k: int = 1) -> Any:
        """Return archdesc from mixed operator."""
        cname = self.candidate_names()
        w = F.softmax(self.alpha().detach(), dim=-1)
        _, cand_idx = torch.topk(w, k)
        desc = [cname[i] for i in cand_idx]
        if desc == []:
            return [None]
        return desc


@register
class IndexMixedOp(MixedOp):
    """Mixed operator controlled by index."""

    def __init__(self, candidates: OrderedDict, arch_param: Optional[Categorical] = None) -> None:
        if arch_param is None:
            arch_param = Categorical(list(candidates.keys()))
        super().__init__(candidates, arch_param)
        self.last_samples = list(range(len(self.candidates())))

    def alpha(self) -> Tensor:
        """Return architecture parameter value."""
        alpha = torch.zeros(len(self.candidates()))
        alpha[self.arch_param.index()] = 1.0
        return alpha

    def forward(self, *args, **kwargs) -> Tensor:
        """Compute MixedOp output."""
        cands = self.candidates()
        smp = self.arch_param.index()
        if self.training:
            self.swap_ops([smp])
        self.last_samples = [smp]
        return cands[smp](*args, **kwargs)

    def swap_ops(self, samples: List[int]) -> None:
        """Remove unused candidates from computation graph."""
        cands = self.candidates()
        for i in self.last_samples:
            if i in samples:
                continue
            for p in cands[i].parameters():
                if p.grad_fn is not None:
                    continue
                p.requires_grad = False
                p.grad = None
        for i in samples:
            for p in cands[i].parameters():
                if p.grad_fn is not None:
                    continue
                p.requires_grad_(True)

    def to_arch_desc(self, *args, **kwargs) -> str:
        """Return archdesc from mixed operator."""
        return self.arch_param_value()
