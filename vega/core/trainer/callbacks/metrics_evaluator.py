# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ProgressLogger call defination."""
from .callbacks import Callback
from copy import deepcopy
import torch
import numpy as np


class MetricsEvaluator(Callback):
    """Callback that shows the progress of evaluating metrics."""

    def __init__(self, mode='MAX', key=None):
        """Initialize MetricsUpdator callback."""
        self.perfs_cmp_mode = mode
        self.perfs_cmp_key = key

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.do_validation = self.params.get('do_validation', False)
        self.cur_loss = None
        self.loss_avg = None
        self.cur_train_perfs = None
        self.best_train_perfs = None
        self.cur_valid_perfs = None
        self.best_valid_perfs = None
        self.best_valid_changed = False
        self.summary_perfs = None

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
        input, target = self.train_batch
        if self.trainer.cfg.is_detection_trainer:
            self.cur_loss = logs['loss']
            self.loss_avg = self.cur_loss
        else:
            batch_size = input.size(0)
            self.cur_loss = logs['loss']
            self.loss_avg = self._average_loss(batch_size, self.cur_loss)
        output = logs['train_batch_output']
        if self.train_metrics is not None:
            metrics_names = self.train_metrics.names
            self.train_metrics(output, target)
            if "is_detection_trainer" in logs.keys() and logs["is_detection_trainer"]:
                metrics_vals = torch.from_numpy(np.array([[0]]))
            else:
                metrics_vals = self.train_metrics.results
            metrics_results = dict(zip(metrics_names, metrics_vals))
            logs.update({'cur_loss': self.cur_loss, 'loss_avg': self.loss_avg,
                         'train_step_metrics': metrics_results})
        else:
            logs.update({'cur_loss': self.cur_loss, 'loss_avg': self.loss_avg})

    def before_valid_step(self, batch_index, logs=None):
        """Be called before a batch validation."""
        self.valid_batch = logs['valid_batch']

    def after_valid_step(self, batch_index, logs=None):
        """Be called after each batch of validation."""
        if self.do_validation and self.valid_metrics is not None:
            input, target = self.valid_batch
            output = logs['valid_batch_output']
            metrics_names = self.valid_metrics.names
            self.valid_metrics(output, target)
            metrics_vals = self.valid_metrics.results
            metrics_results = dict(zip(metrics_names, metrics_vals))
            logs.update({'valid_step_metrics': metrics_results})

    def after_valid(self, logs=None):
        """Be called after validation."""
        if self.do_validation and self.valid_metrics is not None:
            metrics_names = self.valid_metrics.names
            # Get the summary of valid metrics
            metrics_results = self.valid_metrics.results
            self.cur_valid_perfs = dict(zip(metrics_names, metrics_results))
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
                         'best_valid_perfs_changed': self.best_valid_changed})

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        self.summary_perfs = logs.get('summary_perfs', {})
        self.summary_perfs.update({'loss_avg': self.loss_avg})
        if self.train_metrics is not None:
            metrics_names = self.train_metrics.names
            # Get the summary of train metrics
            if self.params["is_detection_trainer"]:
                metrics_results = torch.from_numpy(np.array([[0]]))
            else:
                metrics_results = self.train_metrics.results
            self.cur_train_perfs = dict(zip(metrics_names, metrics_results))
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
                                       'best_valid_perfs_changed': self.best_valid_changed})
        logs.update({'summary_perfs': self.summary_perfs})

    def _update_best_perfs(self, cur_perfs, best_perfs):
        best_changed = False
        best_val = None
        cur_val = None
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
