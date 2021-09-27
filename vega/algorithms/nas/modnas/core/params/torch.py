# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Torch tensor parameter."""
import torch
from .base import Param
from modnas.registry.params import register
from modnas.core.param_space import ParamSpace
from torch.nn.parameter import Parameter
from typing import Optional, Callable


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
