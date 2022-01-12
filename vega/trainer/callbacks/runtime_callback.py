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

"""Report callback defination."""
from vega.common import ClassFactory, ClassType
from vega.metrics.runtime_estimate import RuntimeEstimator
from .callback import Callback


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
