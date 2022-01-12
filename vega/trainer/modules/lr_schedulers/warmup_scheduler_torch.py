# -*- coding: utf-8 -*-

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

"""Basic Warm up lr scheduler.

Example:
    >>> # in yml file `trainer` part
    >>> # use WarmupScheduler and MultiStepLR as after_scheduler
    >>> lr_scheduler:
    >>>     type: WarmupScheduler
    >>>     by_epoch: False
    >>>     params:
    >>>         warmup_type: linear | constant | exp
    >>>         warmup_iters: 20
    >>>         warmup_ratio: 0.1
    >>>         after_scheduler_config:
    >>>             by_epoch: False
    >>>             type: MultiStepLR
    >>>             params:
    >>>                 milestones: [60, 120]
    >>>                 gamma: 0.5

"""

from vega.common import ClassFactory, ClassType
from torch.optim.lr_scheduler import _LRScheduler
from vega.trainer.modules.lr_schedulers import LrScheduler


@ClassFactory.register(ClassType.LR_SCHEDULER)
class WarmupScheduler(_LRScheduler):
    """Multiple Step learning rate with warm up.

    :param milestones: list of decay epochs
    :type milestones: list of int
    :param decay_rates: list of decay rates
    :type decay_rates: list of float
    :param warmup: whether to warm up
    :type warmup: bool
    :param epoch_steps: steps in one epoch
    :type epoch_steps: int
    """

    def __init__(self,
                 optimizer,
                 warmup_type='linear',
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 after_scheduler_config=None,
                 **kwargs):
        """Init WarmupScheduler."""
        if warmup_type is not None:
            if not isinstance(warmup_iters, int) or warmup_iters <= 0:
                raise ValueError('"warmup_iters" must be a positive integer')
            if not isinstance(warmup_ratio, float) or warmup_ratio <= 0 or warmup_ratio > 1.0:
                raise ValueError('"warmup_ratio" must be in range (0,1]')
        self.optimizer = optimizer
        self.warmup_type = warmup_type
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.after_scheduler_config = after_scheduler_config
        self.current_iters = 0
        self.warmup_finished = False
        super(WarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        """Get lr."""
        if self.warmup_finished:
            return self.after_scheduler.get_lr()

        if self.warmup_type == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.base_lrs]
        elif self.warmup_type == 'linear':
            k = (1 - self.current_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.base_lrs]
        elif self.warmup_type == 'exp':
            k = self.warmup_ratio ** (1 - self.current_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.base_lrs]
        return warmup_lr

    def step(self, epoch=None):
        """Step forward for current scheduler."""
        if self.warmup_finished:
            self.after_scheduler.step(epoch)
            return

        if epoch is not None:
            self.current_iters = epoch
            warmup_lr = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr

            if epoch >= self.warmup_iters:
                self.warmup_finished = True
                self.after_scheduler = LrScheduler(self.after_scheduler_config)(self.optimizer)
                self.by_epoch = self.after_scheduler.by_epoch
