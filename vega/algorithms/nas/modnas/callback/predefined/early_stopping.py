# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Early stopping."""
from modnas.registry.callback import register
from ..base import CallbackBase


@register
class EarlyStopping(CallbackBase):
    """Early stopping callback class."""

    priority = -10

    def __init__(self, threshold=10):
        super().__init__({
            'before:EstimBase.run': self.reset,
            'after:EstimBase.step_done': self.on_step_done,
            'after:EstimBase.run_epoch': self.on_epoch,
        })
        self.threshold = threshold
        self.last_opt = -1
        self.stop = False

    def reset(self, estim, optim):
        """Reset callback states."""
        self.last_opt = -1
        self.stop = False

    def on_step_done(self, ret, estim, params, value, arch_desc=None):
        """Check early stop in each step."""
        ret = ret or {}
        if ret.get('is_opt'):
            self.last_opt = -1
        return ret

    def on_epoch(self, ret, estim, optim, epoch, tot_epochs):
        """Check early stop in each epoch."""
        self.last_opt += 1
        if self.last_opt >= self.threshold:
            ret = ret or {}
            self.logger.info('Early stopped: {}'.format(self.last_opt))
            ret['stop'] = True
        return ret
