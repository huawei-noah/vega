# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Basic Warm up lr scheduler.

Example:
    >>> # in yml file `trainer` part
    >>> # use WarmupScheduler and MultiStepLR as after_scheduler
    >>> lr_scheduler:
    >>>     type: WarmupScheduler
    >>>     by_epoch: False
    >>>     warmup_type: linear
    >>>     warmup_iters: 20
    >>>     warmup_ratio: 0.1
    >>>     after_scheduler_config:
    >>>         type: MultiStepLR
    >>>         milestones: [60, 120]
    >>>         gamma: 0.5
    >>>     after_scheduler_by_epoch: True

"""
import copy
import importlib
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.LR_SCHEDULER)
class WarmupScheduler(_LRScheduler):
    """An WarmupScheduler, inherit from torch.optim.lr_scheduler._LRScheduler.

    :param optimizer: Description of parameter `optimizer`.
    :type optimizer: torch.optim.optimizer
    :param by_epoch: step by epoch or by iter.
    :type by_epoch: bool
    :param warmup_type: one of ['constant', 'linear', 'exp'], default is None.
    :type warmup_type: str
    :param warmup_iters: how many iters need to warm up.
    :type warmup_iters: int
    :param warmup_ratio: target ratio of warm up lr of base lr.
    :type warmup_ratio: float
    :param after_scheduler_config: the configs of lr_scheduler to exec after warm up.
    :type after_scheduler_config: dict
    :param after_scheduler_by_epoch: step by epoch or by iter for after_scheduler.
    :type after_scheduler_by_epoch: bool
    :param **kwargs: Description of parameter `**kwargs`.
    :type **kwargs: type
    """

    def __init__(self,
                 optimizer,
                 by_epoch=True,
                 warmup_type=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 after_scheduler_config=None,
                 after_scheduler_by_epoch=True,
                 **kwargs):
        """Init WarmupScheduler."""
        # validate the "warmup" argument
        if warmup_type is not None:
            if warmup_type not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    '"{}" is not a supported type for warming up, valid types'
                    ' are "constant" and "linear"'.format(warmup_type))
        if warmup_type is not None:
            if not isinstance(warmup_iters, int) or warmup_iters <= 0:
                raise ValueError('"warmup_iters" must be a positive integer')
            if not isinstance(warmup_ratio, float) or warmup_ratio <= 0 or warmup_ratio > 1.0:
                raise ValueError('"warmup_ratio" must be in range (0,1]')
        self.optimizer = optimizer
        self.by_epoch = by_epoch
        self.warmup_type = warmup_type
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.after_scheduler_config = after_scheduler_config
        self.after_scheduler_by_epoch = after_scheduler_by_epoch
        self.after_scheduler = None
        self.warmup_finished = False
        self.current_iters = 0
        self._init_after_scheduler()
        # init _LRScheduler
        super(WarmupScheduler, self).__init__(optimizer)

    def _init_after_scheduler(self):
        """Init after_scheduler with after_scheduler_config."""
        if isinstance(self.after_scheduler_config, dict):
            scheduler_config = copy.deepcopy(self.after_scheduler_config)
            print("after_scheduler_config: {}".format(scheduler_config))
            scheduler_name = scheduler_config.pop('type')
            if ClassFactory.is_exists(ClassType.LR_SCHEDULER, scheduler_name):
                scheduler_class = ClassFactory.get_cls(ClassType.LR_SCHEDULER,
                                                       scheduler_name)
            else:
                scheduler_class = getattr(importlib.import_module('torch.optim.lr_scheduler'),
                                          scheduler_name)
            self.after_scheduler = scheduler_class(self.optimizer,
                                                   **scheduler_config)

    def get_lr(self):
        """Get lr."""
        if self.warmup_type is None or self.current_iters > self.warmup_iters:
            if self.after_scheduler is not None:
                if not self.warmup_finished:
                    self.after_scheduler.base_lrs = [base_lr * self.warmup_ratio for base_lr in self.base_lrs]
                    self.warmup_finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.warmup_ratio for base_lr in self.base_lrs]
        if self.warmup_type == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.base_lrs]
        elif self.warmup_type == 'linear':
            k = (1 - self.current_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.base_lrs]
        elif self.warmup_type == 'exp':
            k = self.warmup_ratio**(1 - self.current_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.base_lrs]
        return warmup_lr

    def step(self, epoch=None):
        """Step forward for current scheduler."""
        iter = epoch
        if self.warmup_finished:
            if not self.after_scheduler_by_epoch:
                iter = None
            if self.after_scheduler is not None:
                self.after_scheduler.step(iter)
            else:
                return super(WarmupScheduler, self).step(iter)
        else:
            if iter is None or not self.by_epoch:
                iter = None
                self.current_iters = self.current_iters + 1
                self.last_epoch = self.current_iters
            else:
                self.last_epoch = iter if iter != 0 else 1
                self.current_iters = self.last_epoch
            if self.warmup_type is not None and self.current_iters <= self.warmup_iters:
                warmup_lr = self.get_lr()
                for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                    param_group['lr'] = lr
            elif self.after_scheduler is not None:
                self.after_scheduler.step(iter)
