# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Network module traversal metrics."""
from ..base import MetricsBase
from modnas.registry.metrics import register, build
from modnas.arch_space.mixed_ops import MixedOp


@register
class MixedOpTraversalMetrics(MetricsBase):
    """Mixed operator traversal metrics class."""

    def __init__(self, metrics):
        super().__init__()
        self.metrics = build(metrics)

    def __call__(self, estim):
        """Return metrics output."""
        mt = 0
        for m in estim.model.mixed_ops():
            for p, op in zip(m.prob(), m.candidates()):
                mt = mt + self.metrics(op) * p
        return mt


@register
class ModuleTraversalMetrics(MetricsBase):
    """Module traversal metrics class."""

    def __init__(self, metrics):
        super().__init__()
        self.metrics = build(metrics)

    def __call__(self, estim):
        """Return metrics output."""
        mt = 0
        for m in estim.model.modules():
            if not isinstance(m, MixedOp):
                mt = mt + self.metrics(m)
            else:
                for p, op in zip(m.prob(), m.candidates()):
                    mt = mt + self.metrics(op) * p
        return mt
