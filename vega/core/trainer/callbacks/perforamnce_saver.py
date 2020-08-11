# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""PerformanceSaver callback defination."""
import logging
from .callback import Callback
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class PerformanceSaver(Callback):
    """Callback that saves the evaluated Performance."""

    def __init__(self, best=True, after_epoch=True, after_train=True):
        """Construct a Performance callback."""
        super(Callback, self).__init__()
        self.save_best = best
        self.save_after_epoch = after_epoch
        self.save_after_train = after_train
        self.priority = 250

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.is_chief = self.params['is_chief']
        self.do_validation = self.params['do_validation']
        self.summary_perfs = None
        self.step_name = self.trainer.step_name
        self.worker_id = self.trainer.worker_id

    def after_epoch(self, epoch, logs=None):
        """Be called after the training epoch."""
        logging.debug("train record: saver performance after epoch run successes.")
        self.summary_perfs = logs.get('summary_perfs', {})
        if not (self.is_chief and self.save_after_epoch):
            return
        best_changed = logs.get('best_valid_perfs_changed', False)
        if self.save_best and best_changed:
            pfm = self._get_best_perf(self.summary_perfs)
        else:
            pfm = self._get_cur_perf(self.summary_perfs)
        if pfm:
            if self.summary_perfs.get("gflops"):
                pfm.update({"gflops": self.summary_perfs.get("gflops")})
            if self.summary_perfs.get("kparams"):
                pfm.update({"kparams": self.summary_perfs.get("kparams")})
            self.trainer.performance = pfm

    def _get_cur_perf(self, summary_perfs):
        return summary_perfs.get('cur_valid_perfs', None)

    def _get_best_perf(self, summary_perfs):
        return summary_perfs.get('best_valid_perfs', None)
