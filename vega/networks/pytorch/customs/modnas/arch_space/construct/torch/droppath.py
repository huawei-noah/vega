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

"""DropPath constructor."""
from typing import Optional
import torch
from torch.nn.modules.container import Sequential
from torch.nn.modules.module import Module
from modnas.arch_space.ops import Identity
from modnas.core.event import event_on
from modnas.registry.construct import register
from modnas.arch_space.slot import Slot
from .default import DefaultSlotTraversalConstructor


class DropPath(torch.nn.Module):
    """DropPath module."""

    def __init__(self, prob=0.):
        super().__init__()
        self.drop_prob = prob

    def extra_repr(self):
        """Return extra representation string."""
        return 'prob={}, inplace'.format(self.drop_prob)

    def forward(self, x):
        """Return operator output."""
        if self.training and self.drop_prob > 0.:
            keep_prob = 1. - self.drop_prob
            mask = torch.FloatTensor(x.size(0), 1, 1, 1).to(device=x.device).bernoulli_(keep_prob)
            x.div_(keep_prob).mul_(mask)
        return x


def _apply_drop_prob(module, prob):
    for m in module.modules():
        if isinstance(m, DropPath):
            m.drop_prob = prob


def _parse_drop_prob(drop_prob):
    if isinstance(drop_prob, (tuple, list)):
        return drop_prob[0], drop_prob[1]
    else:
        return 0, drop_prob


@register
class DropPathConverter():
    """Constructor that applies DropPath on a single module."""

    def __init__(self, drop_prob=0.1):
        self.min_drop_prob, self.max_drop_prob = _parse_drop_prob(drop_prob)

    def __call__(self, module):
        """Run constructor."""
        def drop_prob_update(*args, epoch=None, tot_epochs=None, **kwargs):
            _apply_drop_prob(module, self.max_drop_prob * epoch / tot_epochs)

        event_on('before:TrainerBase.train_epoch', drop_prob_update)
        if module is None or isinstance(module, Identity):
            return module
        return torch.nn.Sequential(module, DropPath(self.min_drop_prob))


@register
class DropPathConstructor(DefaultSlotTraversalConstructor):
    """Constructor that applies DropPath on Slot modules."""

    def __init__(self, *args, drop_prob=0.1, skip_exist=False, **kwargs) -> None:
        super().__init__(*args, skip_exist=skip_exist, **kwargs)
        self.min_drop_prob, self.max_drop_prob = _parse_drop_prob(drop_prob)
        self.transf = DropPathTransformer()

    def __call__(self, model: Module) -> Module:
        """Run constructor."""
        super().__call__(model)

        def drop_prob_update(*args, epoch=None, tot_epochs=None, **kwargs):
            self.transf.set_prob(self.max_drop_prob * epoch / tot_epochs)
            self.transf(model)

        event_on('before:TrainerBase.train_epoch', drop_prob_update)
        return model

    def convert(self, slot: Slot) -> Optional[Sequential]:
        """Return module with DropPath."""
        ent = slot.get_entity()
        if ent is None or isinstance(ent, Identity):
            return None
        new_ent = torch.nn.Sequential(ent, DropPath(self.min_drop_prob))
        return new_ent


@register
class DropPathTransformer(DefaultSlotTraversalConstructor):
    """Transformer that update DropPath probability."""

    def __init__(self, *args, skip_exist=False, **kwargs) -> None:
        super().__init__(*args, skip_exist=skip_exist, **kwargs)
        self.prob = None

    def set_prob(self, prob: float) -> None:
        """Set DropPath probability."""
        self.prob = prob

    def convert(self, slot: Slot) -> None:
        """Apply DropPath probability on Slot module."""
        ent = slot.get_entity()
        if ent is None:
            return
        _apply_drop_prob(ent, self.prob)
