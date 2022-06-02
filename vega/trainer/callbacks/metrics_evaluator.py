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

"""ProgressLogger call defination."""

from copy import deepcopy
from vega.common import ClassFactory, ClassType
from .callback import Callback


@ClassFactory.register(ClassType.CALLBACK)
class MetricsEvaluator(Callback):
    """Callback that shows the progress of evaluating metrics."""

    def __init__(self):
        """Initialize MetricsEvaluator callback."""
        super(MetricsEvaluator, self).__init__()
        self.priority = 230

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.do_validation = self.trainer.do_validation
        self.cur_loss = None
        self.loss_avg = None
        self.cur_train_perfs = None
        self.best_train_perfs = None
        self.cur_valid_perfs = None
        self.best_valid_perfs = None
        self.best_valid_changed = False
        self.summary_perfs = None
        self.train_metrics = None
        self.perfs_cmp_mode = self.trainer.config.perfs_cmp_mode
        self.perfs_cmp_key = self.trainer.config.perfs_cmp_key
        # get_train_metric_after_epoch: detector or no need to get train_metrics after epoch
        self.get_train_metric_after_epoch = self.trainer.config.get_train_metric_after_epoch

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        self.train_metrics = self.trainer.train_metrics
        self.valid_metrics = self.trainer.valid_metrics
        self.counted_steps = 0
        self.total_loss = 0
        if self.train_metrics is not None:
            self.train_metrics.reset()
        if self.do_validation and self.valid_metrics is not None:
            self.valid_metrics.reset()

    def before_train_step(self, batch_index, logs=None):
        """Be called before a batch training."""
        self.train_batch = logs['train_batch']

    def after_train_step(self, batch_index, logs=None):
        """Be called after each train batch."""
        if isinstance(self.train_batch, list) and isinstance(self.train_batch[0], dict):
            input, target = self.train_batch, None
        else:
            input, target = self.train_batch
        if isinstance(logs['lr'], list):
            self.lr = logs['lr'][0]
        else:
            self.lr = logs['lr']
        if self.trainer.config.is_detection_trainer:
            self.cur_loss = logs['loss']
            self.loss_avg = self.cur_loss
        else:
            if isinstance(input, dict):
                batch_size = 1
            elif isinstance(input, list):
                batch_size = len(input)
            else:
                batch_size = input.size(0)
            self.cur_loss = logs['loss']
            self.loss_avg = self._average_loss(batch_size, self.cur_loss)
        logs.update({'cur_loss': self.cur_loss, 'loss_avg': self.loss_avg, 'lr': self.lr})

    def before_valid_step(self, batch_index, logs=None):
        """Be called before a batch validation."""
        self.valid_batch = logs['valid_batch']

    def after_valid_step(self, batch_index, logs=None):
        """Be called after each batch of validation."""
        if self.do_validation and self.valid_metrics is not None:
            is_dict = isinstance(self.valid_batch, list) and \
                len(self.valid_batch) == 2 and \
                isinstance(self.valid_batch[0], dict) and \
                isinstance(self.valid_batch[1], dict)
            if is_dict:
                target = self.valid_batch[1]
                output = logs["valid_batch_output"]
                self.valid_metrics(**output, **target)
                return
            elif isinstance(self.valid_batch, list) and isinstance(self.valid_batch[0], dict):
                target = self.valid_batch
            else:
                target = self.valid_batch[1]
            output = logs['valid_batch_output']
            self.valid_metrics(output, target)

    def after_valid(self, logs=None):
        """Be called after validation."""
        if self.do_validation and self.valid_metrics is not None:
            # Get the summary of valid metrics
            metrics_results = self.valid_metrics.results
            if hasattr(self.trainer, "_average_metrics"):
                metrics_results = self.trainer._average_metrics(metrics_results)
            if 'loss' in metrics_results:
                metrics_results.pop('loss')
            if 'global_step' in metrics_results:
                metrics_results.pop('global_step')
            self.cur_valid_perfs = metrics_results
            logs.update({'cur_valid_perfs': self.cur_valid_perfs})
            # update best valid perfs based on current valid valid_perfs
            if self.best_valid_perfs is None:
                self.best_valid_changed = True
                self.best_valid_perfs = self.cur_valid_perfs
            else:
                self.best_valid_changed = self._update_best_perfs(self.cur_valid_perfs,
                                                                  self.best_valid_perfs)
            logs.update({'cur_valid_perfs': self.cur_valid_perfs,
                         'best_valid_perfs': self.best_valid_perfs,
                         'best_changed': self.best_valid_changed})

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        self.summary_perfs = logs.get('summary_perfs', {})
        self.summary_perfs.update({'loss_avg': self.loss_avg})
        if self.train_metrics is not None and self.get_train_metric_after_epoch:
            # Get the summary of train metrics
            metrics_results = self.train_metrics.results
            self.cur_train_perfs = metrics_results
            # update best train perfs based on current train perfs
            if self.best_train_perfs is None:
                self.best_train_perfs = deepcopy(self.cur_train_perfs)
            else:
                self._update_best_perfs(self.cur_train_perfs,
                                        self.best_train_perfs)
            self.summary_perfs.update({'cur_train_perfs': self.cur_train_perfs,
                                       'best_train_perfs': self.best_train_perfs})
        if self.do_validation and self.valid_metrics is not None:
            self.summary_perfs.update({'cur_valid_perfs': self.cur_valid_perfs,
                                       'best_valid_perfs': self.best_valid_perfs,
                                       'best_changed': self.best_valid_changed})

        logs.update({'summary_perfs': self.summary_perfs})

    def after_train(self, logs=None):
        """Be called before training."""
        self.after_epoch(self.trainer.epochs, logs)

    def _update_best_perfs(self, cur_perfs, best_perfs):
        best_changed = False
        if self.perfs_cmp_key is None:
            # Select the first kye as default for comparsion
            self.perfs_cmp_key = list(cur_perfs.keys())[0]
        # Get the values for comparsion based on key
        if isinstance(best_perfs[self.perfs_cmp_key], list):
            best_val = best_perfs[self.perfs_cmp_key][0]
            cur_val = cur_perfs[self.perfs_cmp_key][0]
        else:
            best_val = best_perfs[self.perfs_cmp_key]
            cur_val = cur_perfs[self.perfs_cmp_key]
        # Store the perfs after comparison based on mode
        if self.perfs_cmp_mode is None:
            self.perfs_cmp_mode = self.valid_metrics.objectives.get(self.perfs_cmp_key)
        if self.perfs_cmp_mode == 'MAX':
            if cur_val > best_val:
                best_perfs.update(deepcopy(cur_perfs))
                best_changed = True
        elif self.perfs_cmp_mode == 'MIN':
            if cur_val < best_val:
                best_perfs.update(deepcopy(cur_perfs))
                best_changed = True
        else:
            best_perfs.update(deepcopy(cur_perfs))
            best_changed = True
        return best_changed

    def _average_loss(self, batch_size, cur_loss):
        self.counted_steps += batch_size
        self.total_loss += cur_loss * batch_size
        averaged_loss = self.total_loss / self.counted_steps
        return averaged_loss
