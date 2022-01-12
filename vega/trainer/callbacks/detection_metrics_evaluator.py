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

"""DetectionMetricsEvaluator call defination."""

from vega.trainer.callbacks.metrics_evaluator import MetricsEvaluator
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class DetectionMetricsEvaluator(MetricsEvaluator):
    """Callback that shows the progress of detection evaluating metrics."""

    def __init__(self, *args, **kwargs):
        """Initialize class."""
        super().__init__(*args, **kwargs)

    def before_train(self, logs=None):
        """Be called before the training process."""
        super().before_train(logs)
        self.step_count_during_train_period = 0
        self.loss_sum_during_train_period = 0

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        super().before_epoch(epoch, logs)
        self.loss_sum_during_epoch_period = 0
        self.step_count_during_epoch_period = 0

    def after_train_step(self, batch_index, logs=None):
        """Be called after each train batch."""
        input, _ = self.train_batch
        batch_size = input.size(0)
        self.cur_loss = logs['loss']
        self.loss_avg = self._average_loss_during_train_period(batch_size, self.cur_loss)
        logs.update({'cur_loss': self.cur_loss, 'loss_avg': self.loss_avg})

    def after_valid_step(self, batch_index, logs=None):
        """Be called after each batch of validation."""
        if self.trainer.do_validation and self.valid_metrics is not None:
            _, target = self.valid_batch
            output = logs['valid_batch_output']
            self.valid_metrics(output, target)

    def _average_loss_during_epoch_period(self, batch_size, cur_loss):
        self.loss_sum_during_epoch_period = self.loss_sum_during_epoch_period + cur_loss * batch_size
        self.step_count_during_epoch_period = self.step_count_during_epoch_period + batch_size
        avg_loss = self.loss_sum_during_epoch_period / self.step_count_during_epoch_period
        return avg_loss

    def _average_loss_during_train_period(self, batch_size, cur_loss):
        self.step_count_during_train_period = self.step_count_during_train_period + batch_size
        self.loss_sum_during_train_period = self.loss_sum_during_train_period + cur_loss * batch_size
        avg_loss = self.loss_sum_during_train_period / self.step_count_during_train_period
        return avg_loss
