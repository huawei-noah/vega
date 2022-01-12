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

"""Cosine annealing lr scheduler."""
import math
import numpy as np
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LR_SCHEDULER)
class MultiStepLR():
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

    def __init__(self, optimizer=None, milestones=None, gamma=0.1):
        """Initialize."""
        super(MultiStepLR, self).__init__()
        self.milestones = milestones
        self.gamma = gamma

    def __call__(self, base_lr, global_step, total_epoch):
        """Call lr scheduler class."""
        lr_each_step = []
        decay_step_index = [int(global_step * (self.milestones[i] / total_epoch)) for i in
                            range(len(self.milestones))]

        for i in range(global_step):
            if i < decay_step_index[0]:
                lr_each_step.append(base_lr)
            elif i < decay_step_index[min(1, len(self.milestones) - 1)]:
                lr_each_step.append(base_lr * self.gamma)
            elif i < decay_step_index[min(2, len(self.milestones) - 1)]:
                lr_each_step.append(base_lr * self.gamma * self.gamma)
            else:
                lr_each_step.append(base_lr * self.gamma * self.gamma * self.gamma)
        lr_each_step = np.array(lr_each_step).astype(np.float32)
        return lr_each_step


@ClassFactory.register(ClassType.LR_SCHEDULER)
class StepLR():
    """StepLR learning rate.

    :param step_size: the epoch interval to decay lr
    :type step_size: int
    :param gamma: the decay rate
    :type gamma: float
    """

    def __init__(self, optimizer=None, step_size=None, gamma=0.1):
        """Initialize."""
        super(StepLR, self).__init__()
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, base_lr, global_step, total_epoch):
        """Call lr scheduler class."""
        lr_each_step = []
        cur_lr = base_lr
        for i in range(global_step):
            cur_epoch = int(i // (global_step / total_epoch))
            if cur_epoch > 0 and i % cur_epoch == 0:
                cur_lr = base_lr * self.gamma ** (cur_epoch // self.step_size)
            lr = cur_lr
            lr_each_step.append(lr)
        return lr_each_step


@ClassFactory.register(ClassType.LR_SCHEDULER)
class CosineAnnealingLR():
    """CosineAnnealingLR learning rate.

    :param T_max: the half period of cosine function
    :type T_max: int
    :param eta_min: the minimal learning rate
    :type eta_min: float
    """

    def __init__(self, optimizer=None, T_max=None, eta_min=0):
        """Initialize."""
        super(CosineAnnealingLR, self).__init__()
        self.T_max = T_max
        self.eta_min = eta_min

    def __call__(self, base_lr, global_step, total_epoch):
        """Call lr scheduler class."""
        lr_each_step = []
        cur_lr = base_lr
        for cur_step in range(global_step):
            cur_epoch = int(cur_step // (global_step / total_epoch))
            if cur_epoch > 0 and cur_step % cur_epoch == 0:
                cur_lr = self.eta_min + (base_lr - self.eta_min) * (1. + math.cos(math.pi * cur_epoch / self.T_max)) / 2
            lr = cur_lr
            lr_each_step.append(lr)
        return lr_each_step


@ClassFactory.register(ClassType.LR_SCHEDULER)
class PolyLR():
    """Applies polynomial decay to generate learning rate array."""

    def __init__(self, optimizer=None, lr_max=0.1):
        super(PolyLR, self).__init__()
        self.lr_max = lr_max

    def __call__(self, base_lr, global_step, total_epoch):
        """Call lr scheduler class."""
        lr_each_step = []
        for cur_step in range(global_step):
            base = 1 - cur_step / global_step
            lr = self.lr_max * base * base
            lr_each_step.append(lr)
        return lr_each_step


@ClassFactory.register(ClassType.LR_SCHEDULER)
class WarmupScheduler():
    """WarmupScheduler learning rate."""

    def __init__(self, optimizer=None, warmup_type="linear", warmup_iters=0, warmup_ratio=0.01,
                 after_scheduler_config=None):
        super(WarmupScheduler, self).__init__()
        self.warmup_type = warmup_type
        self.warmup_iter = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.after_scheduler_config = after_scheduler_config
        self.after_scheduler_cls = ClassFactory.get_cls(ClassType.LR_SCHEDULER, self.after_scheduler_config.get("type"))
        self.after_scheduler = self.after_scheduler_cls(optimizer=None, **(self.after_scheduler_config.get("params")))

    def __call__(self, base_lr, global_step, total_epoch):
        """Call lr scheduler class."""
        if self.warmup_type == "constant":
            warmup_lrs = [base_lr * self.warmup_ratio] * self.warmup_iter
        elif self.warmup_type == "linear":
            warmup_lrs = [base_lr * (i + 1) / self.warmup_iter for i in range(self.warmup_iter)]

        remaining_iters = global_step - self.warmup_iter
        remaining_epochs = remaining_iters / global_step * total_epoch
        after_warmup_lrs = self.after_scheduler(base_lr, remaining_iters, remaining_epochs)
        return warmup_lrs + list(after_warmup_lrs)
