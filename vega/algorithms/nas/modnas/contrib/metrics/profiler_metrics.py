# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Local network hardware performance profiler metrics."""
import time
import torch
from modnas.registry.metrics import register
from modnas.metrics.base import MetricsBase


@register
class LocalProfilerMetrics(MetricsBase):
    """Local network hardware performance profiler metrics class."""

    def __init__(self, device=None, rep=50, warmup=10):
        super().__init__()
        self.rep = rep
        self.warmup = warmup
        self.device = device

    def __call__(self, node):
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
