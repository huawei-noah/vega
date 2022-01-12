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

"""Network module traversal metrics."""
from modnas.registry.metrics import register, build
from modnas.arch_space.mixed_ops import MixedOp
from ..base import MetricsBase


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
