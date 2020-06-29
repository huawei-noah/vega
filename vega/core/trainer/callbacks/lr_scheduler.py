# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""LearningRateSchduler callback Defination."""
from .callbacks import Callback
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class LearningRateScheduler(Callback):
    """Callback that adjust the learning rate during training."""

    def __init__(self, call_point="BEFORE_EPOCH"):
        """Init LearningRateSchduler callback."""
        self.call_point = call_point

    def before_train(self, logs=None):
        """Be called before training."""
        self.lr_scheduler = self.trainer.lr_scheduler

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoch."""
        if self.call_point == "BEFORE_EPOCH" and self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch=epoch)

    def after_epoch(self, epoch, logs=None):
        """Be called before each epoch."""
        if self.call_point == "AFTER_EPOCH" and self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch=epoch)
