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

"""Early stopping."""
from collections import OrderedDict
from typing import Any, Dict, Optional
from modnas.registry.callback import register
from modnas.estim.base import EstimBase
from modnas.optim.base import OptimBase
from ..base import CallbackBase


_ret_type = Optional[Dict[str, Any]]


@register
class EarlyStopping(CallbackBase):
    """Early stopping callback class."""

    priority = -10

    def __init__(self, threshold: int = 10) -> None:
        super().__init__({
            'before:EstimBase.run': self.reset,
            'after:EstimBase.step_done': self.on_step_done,
            'after:EstimBase.run_epoch': self.on_epoch,
        })
        self.threshold = threshold
        self.last_opt = -1
        self.stop = False

    def reset(self, estim: EstimBase, optim: OptimBase) -> None:
        """Reset callback states."""
        self.last_opt = -1
        self.stop = False

    def on_step_done(
        self, ret: _ret_type, estim: EstimBase, params: OrderedDict, value: float, arch_desc: Optional[Any] = None
    ) -> _ret_type:
        """Check early stop in each step."""
        ret = ret or {}
        if ret.get('is_opt'):
            self.last_opt = -1
        return ret

    def on_epoch(self, ret: _ret_type, estim: EstimBase, optim: OptimBase, epoch: int, tot_epochs: int) -> _ret_type:
        """Check early stop in each epoch."""
        self.last_opt += 1
        if self.last_opt >= self.threshold:
            ret = ret or {}
            self.logger.info('Early stopped: {}'.format(self.last_opt))
            ret['stop'] = True
        return ret
