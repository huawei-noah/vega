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


def _default_tensor_sampler(shape, init_ratio=1e-3):
    return torch.nn.Parameter(init_ratio * torch.randn(shape))


@register
class TorchTensor(Param):
    """Torch tensor parameter class."""

    TYPE = 'T'

    def __init__(self, shape, sampler=None, name=None, space=None, on_update=None):
        super().__init__(name, space, on_update)
        self.sample = _default_tensor_sampler if sampler is None else sampler
        self.shape = shape
        self.val = self.sample(self.shape)
        self._length = None

    def extra_repr(self):
        """Return extra representation string."""
        return 'shape={}'.format(self.shape)

    def is_valid(self, value):
        """Return if the value is valid."""
        return isinstance(value, torch.Tensor)

    def value(self):
        """Return parameter value."""
        if self.val is None:
            self.val = self.sample(self.shape)
        return self.val

    def set_value(self, value):
        """Set parameter value."""
        self.val = value
