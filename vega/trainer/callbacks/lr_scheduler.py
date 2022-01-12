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

"""LearningRateSchduler callback Defination."""
from vega.common import ClassFactory, ClassType
from .callback import Callback


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
            step = self.trainer.batch_num_train * self.epoch + batch_index
            self.lr_scheduler.step(epoch=step)
