# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""PerformanceSaver callback defination."""
from .callbacks import Callback
from copy import deepcopy
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class PerformanceSaver(Callback):
    """Callback that saves the evaluated Performance."""

    def __init__(self, best=True, after_epoch=True, after_train=True):
        """Construct a Performance callback."""
        self.save_best = best
        self.save_after_epoch = after_epoch
        self.save_after_train = after_train

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.is_chief = self.params['is_chief']
        self.do_validation = self.params['do_validation']
        self.summary_perfs = None

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        self.summary_perfs = logs.get('summary_perfs', {})
        if self.is_chief and self.save_after_epoch:
            if self.do_validation:
                if self.save_best:
                    best_changed = logs.get('best_valid_perfs_changed', False)
                    if best_changed:
                        best_perf = self._get_best_perf(self.summary_perfs)
                        self.trainer._save_performance(best_perf)
                else:
                    cur_perf = self._get_cur_perf(self.summary_perfs)
                    self.trainer._save_performance(cur_perf)
            else:
                self.trainer._save_performance(self.summary_perfs)

    def after_train(self, logs=None):
        """Be called after the training process."""
        if self.is_chief and self.save_after_train:
            if self.do_validation:
                best_perf = self._get_best_perf(self.summary_perfs)
                self.trainer._save_performance(best_perf)
            else:
                self.trainer._save_performance(self.summary_perfs)

    def _get_cur_perf(self, summary_perfs):
        cur_valid_perfs = summary_perfs.get('cur_valid_perfs', None)
        first_metric_val = list(cur_valid_perfs.values())[0]
        return first_metric_val

    def _get_best_perf(self, summary_perfs):
        best_valid_perfs = summary_perfs.get('best_valid_perfs', None)
        first_metric_val = list(best_valid_perfs.values())[0]
        return first_metric_val

    def _flatten_summary_perfs(self, summary_perfs):
        # This function for future use
        flatten_summary_perfs = deepcopy(summary_perfs)
        cur_train_perfs = flatten_summary_perfs.pop('cur_train_perfs', None)
        if cur_train_perfs is not None:
            prefix_cur_train_perfs = self._prefix_perfs(
                cur_train_perfs, 'train')
            flatten_summary_perfs.update(prefix_cur_train_perfs)

        best_train_perfs = flatten_summary_perfs.pop('best_train_perfs', None)
        if best_train_perfs is not None:
            prefix_best_train_perfs = self._prefix_perfs(best_train_perfs,
                                                         'train_best')
            flatten_summary_perfs.update(prefix_best_train_perfs)

        cur_valid_perfs = flatten_summary_perfs.pop('cur_valid_perfs', None)
        if cur_valid_perfs is not None:
            prefix_cur_valid_perfs = self._prefix_perfs(
                cur_valid_perfs, 'valid')
            flatten_summary_perfs.update(prefix_cur_valid_perfs)

        best_valid_perfs = flatten_summary_perfs.pop('best_valid_perfs', None)
        if best_valid_perfs is not None:
            prefix_best_valid_perfs = self._prefix_perfs(best_valid_perfs,
                                                         'valid_best')
            flatten_summary_perfs.update(prefix_best_valid_perfs)
        return flatten_summary_perfs

    def _prefix_perfs(self, perfs, prefix):
        prefix_perfs = {}
        for name, val in perfs.items():
            prefix_perfs["{}_{}".format(prefix, name)] = val
        return prefix_perfs
