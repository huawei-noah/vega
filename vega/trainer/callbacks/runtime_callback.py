# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Report callback defination."""
from .callback import Callback
from vega.common import ClassFactory, ClassType
from vega.metrics.runtime_estimate import RuntimeEstimator


@ClassFactory.register(ClassType.CALLBACK)
class RuntimeCallback(Callback):
    """Running time estimate callback."""

    def __init__(self):
        super(RuntimeCallback, self).__init__()
        self.remain_time = dict()
        self.whole_time = dict()
        self.priority = 210

    def before_train(self, logs=None):
        """Define runtime type and mark train start time."""
        epochs = self.trainer.epochs
        self.rt_est = RuntimeEstimator(types='train', max_steps=epochs)
        train_steps = self.trainer.batch_num_train
        self.rt_est.add_runtime_est(type='epoch', max_step=train_steps)
        self.rt_est.mark_start_time('train', step=0)

    def before_epoch(self, epoch, logs=None):
        """Mark epoch start time."""
        self.rt_est.mark_start_time('epoch', step=0)

    def after_epoch(self, epoch, logs=None):
        """Obtain estimated running time after epoch."""
        self.remain_time['train'] = self.rt_est.remaining_time('train', step=epoch + 1)
        using_time = self.rt_est.using_time('train')
        self.whole_time['train'] = self.remain_time['train'] + using_time
        logs.update({'runtime': {'remain_time': self.remain_time,
                                 'whole_time': self.whole_time}})
        self.trainer.runtime = self.whole_time['train']

    def after_train_step(self, batch_index, logs=None):
        """Obtain estimated running time after step."""
        self.remain_time['epoch'] = self.rt_est.remaining_time('epoch', step=batch_index + 1)
        using_time = self.rt_est.using_time('epoch')
        self.whole_time['epoch'] = self.remain_time['epoch'] + using_time

    def after_train(self, logs=None):
        """Restore train time in trainer."""
        if 'train' not in self.whole_time:
            self.after_epoch(0, logs)
        self.trainer.runtime = self.whole_time['train']
