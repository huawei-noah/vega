# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""LearningRateSchduler callback Defination."""
from .callback import Callback
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class LearningRateScheduler(Callback):
    """Callback that adjust the learning rate during training."""

    def __init__(self):
        """Initialize LearningRateScheduler callback."""
        super(LearningRateScheduler, self).__init__()
        self.priority = 260

    def before_train(self, logs=None):
        """Be called before training."""
        self.lr_scheduler = self.trainer.lr_scheduler

    def before_epoch(self, epoch, logs=None):
        """Call before_epoch of the managed callbacks."""
        self.epoch = epoch

    def after_epoch(self, epoch, logs=None):
        """Be called before each epoch."""
        if self.lr_scheduler and self.lr_scheduler.by_epoch:
            self.lr_scheduler.step(epoch=epoch)

    def after_train_step(self, batch_index, logs=None):
        """Call after_train_step of the managed callbacks."""
        if self.lr_scheduler and not self.lr_scheduler.by_epoch:
            step = self.trainer.batch_num_train * self.epoch + self.epoch + batch_index
            self.lr_scheduler.step(epoch=step)
