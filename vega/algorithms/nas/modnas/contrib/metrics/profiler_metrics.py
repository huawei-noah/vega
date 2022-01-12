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

"""Local network hardware performance profiler metrics."""
import time
import torch
from modnas.registry.metrics import register
from modnas.metrics.base import MetricsBase
from typing import Optional
from rasp.profiler.tree import StatTreeNode


@register
class LocalProfilerMetrics(MetricsBase):
    """Local network hardware performance profiler metrics class."""

    def __init__(self, device: Optional[str] = None, rep: int = 50, warmup: int = 10) -> None:
        super().__init__()
        self.rep = rep
        self.warmup = warmup
        self.device = device

    def __call__(self, node: StatTreeNode) -> float:
        """Return metrics output."""
        in_shape = node['in_shape']
        op = node.module
        plist = list(op.parameters())
        if len(plist) == 0:
            last_device = None
        else:
            last_device = plist[0].device
        device = last_device if self.device is None else self.device
        x = torch.randn(in_shape).to(device=device)
        op = op.to(device=device)
        tic = time.perf_counter()
        with torch.no_grad():
            for rep in range(self.warmup + self.rep):
                if rep == self.warmup:
                    tic = time.perf_counter()
                torch.cuda.synchronize()
                op(x)
                torch.cuda.synchronize()
        toc = time.perf_counter()
        lat = 1000. * (toc - tic) / self.rep
        op.to(device=last_device)
        self.logger.debug('local profiler:\tdev: {}\tlat: {:.3f} ms'.format(device, lat))
        return lat
