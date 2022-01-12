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

"""PerformanceSaver callback defination."""
import logging
from vega.common import ClassFactory, ClassType
from .callback import Callback


@ClassFactory.register(ClassType.CALLBACK)
class PerformanceSaver(Callback):
    """Callback that saves the evaluated Performance."""

    def __init__(self, best=True, after_epoch=True, after_train=True):
        """Construct a Performance callback."""
        super(PerformanceSaver, self).__init__()
        self.save_best = best
        self.save_after_epoch = after_epoch
        self.save_after_train = after_train
        self.priority = 250

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.summary_perfs = logs.get('summary_perfs', {})
        self.step_name = self.trainer.step_name
        self.worker_id = self.trainer.worker_id
        pfm = {}
        if not self.summary_perfs.get("flops") is None:
            pfm.update({"flops": self.summary_perfs.get("flops")})
        if not self.summary_perfs.get("params") is None:
            pfm.update({"params": self.summary_perfs.get("params")})
        self.trainer.performance = pfm

    def after_epoch(self, epoch, logs=None):
        """Be called after the training epoch."""
        logging.debug("train record: saver performance after epoch run successes.")
        if not (self.trainer.is_chief and self.save_after_epoch):
            return
        self._update_pfm(logs)

    def after_train(self, logs=None):
        """Be called after training."""
        self._update_pfm(logs)

    def _update_pfm(self, logs):
        self.summary_perfs = logs.get('summary_perfs', {})

        best_changed = self.summary_perfs.get('best_changed', False)
        if self.save_best and best_changed:
            pfm = self._get_best_perf(self.summary_perfs)
            self.trainer.best_performance = pfm
        else:
            pfm = self._get_cur_perf(self.summary_perfs)
        if pfm:
            if not self.summary_perfs.get("flops") is None:
                pfm.update({"flops": self.summary_perfs.get("flops")})
            if not self.summary_perfs.get("params") is None:
                pfm.update({"params": self.summary_perfs.get("params")})
            if not self.summary_perfs.get("latency") is None:
                pfm.update({"latency": self.summary_perfs.get("latency")})
            self.trainer.performance = pfm

    def _get_cur_perf(self, summary_perfs):
        return summary_perfs.get('cur_valid_perfs', None)

    def _get_best_perf(self, summary_perfs):
        return summary_perfs.get('best_valid_perfs', None)
