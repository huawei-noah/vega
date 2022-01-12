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
