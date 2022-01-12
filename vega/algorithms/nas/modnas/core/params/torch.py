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

"""Torch tensor parameter."""

from typing import Optional, Callable
import torch
from torch.nn.parameter import Parameter
from modnas.registry.params import register
from modnas.core.param_space import ParamSpace
from .base import Param


def _default_tensor_sampler(shape: int, init_ratio: float = 1e-3) -> Parameter:
    return torch.nn.Parameter(init_ratio * torch.randn(shape))


@register
class TorchTensor(Param):
    """Torch tensor parameter class."""

    TYPE = 'T'

    def __init__(
        self, shape: int, sampler: Optional[Callable] = None, name: Optional[str] = None,
        space: Optional[ParamSpace] = None, on_update: Optional[Callable] = None
    ) -> None:
        super().__init__(name, space, on_update)
        self.sample = _default_tensor_sampler if sampler is None else sampler
        self.shape = shape
        self.val = self.sample(self.shape)
        self._length = None

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return 'shape={}'.format(self.shape)

    def is_valid(self, value):
        """Return if the value is valid."""
        return isinstance(value, torch.Tensor)

    def value(self) -> Parameter:
        """Return parameter value."""
        if self.val is None:
            self.val = self.sample(self.shape)
        return self.val

    def set_value(self, value: Parameter) -> None:
        """Set parameter value."""
        self.val = value
