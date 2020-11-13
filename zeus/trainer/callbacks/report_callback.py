# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Report callback defination."""
import logging
from .callback import Callback
from zeus.report import Report
from zeus.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class ReportCallback(Callback):
    """Callback that report records."""

    def __init__(self):
        """Initialize ReportCallback callback."""
        super(Callback, self).__init__()
        self.epoch = 0
        self.priority = 280

    def after_valid(self, logs=None):
        """Be called after each epoch."""
        if self.trainer.config.report_on_valid:
            self._broadcast()

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        self.epoch = epoch
        self._broadcast(epoch)

    def after_train(self, logs=None):
        """Close the connection of report."""
        self._broadcast(self.epoch)
        Report().close(self.trainer.step_name, self.trainer.worker_id)

    def _broadcast(self, epoch=None):
        record = Report().receive(self.trainer.step_name, self.trainer.worker_id)
        if self.trainer.config.report_on_epoch:
            record.epoch = self.trainer.epochs
        # todo: remove in FinedGrainedSpace
        if self.trainer.config.codec:
            record.desc = self.trainer.config.codec
        if not record.desc:
            record.desc = self.trainer.model_desc
        record.performance = self.trainer.performance
        record.objectives = self.trainer.valid_metrics.objectives
        if record.performance is not None:
            for key in record.performance:
                if key not in record.objectives:
                    if (key == 'flops' or key == 'params' or key == 'latency'):
                        record.objectives.update({key: 'MIN'})
                    else:
                        record.objectives.update({key: 'MAX'})
        record.model_path = self.trainer.model_path
        record.checkpoint_path = self.trainer.checkpoint_file
        record.weights_file = self.trainer.weights_file
        if self.trainer.runtime is not None:
            record.runtime = self.trainer.runtime
        Report().broadcast(record)
        logging.debug("report_callback record: {}".format(record))
