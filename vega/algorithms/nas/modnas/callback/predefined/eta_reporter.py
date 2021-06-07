# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ETA (remaining time) reporter."""
from modnas.utils import ETAMeter
from modnas.registry.callback import register
from ..base import CallbackBase


@register
class ETAReporter(CallbackBase):
    """ETA reporter class."""

    priority = -1

    def __init__(self):
        super().__init__({
            'before:EstimBase.run': self.init,
            'after:EstimBase.run_epoch': self.report_epoch,
        })
        self.eta_m = None

    def init(self, estim, *args, **kwargs):
        """Initialize ETA meter."""
        tot_epochs = estim.config.get('epochs', 0)
        if tot_epochs < 1:
            return
        self.eta_m = ETAMeter(tot_epochs, estim.cur_epoch)
        self.eta_m.start()

    def report_epoch(self, ret, estim, *args, **kwargs):
        """Report ETA in each epoch."""
        if self.eta_m is None:
            return
        ret = ret or {}
        self.eta_m.step()
        ret['ETA'] = self.eta_m.eta_fmt()
        return ret
