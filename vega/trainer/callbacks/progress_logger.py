# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ProgressLogger call defination."""
import logging
import time
import numpy as np
from collections.abc import Iterable
from .callback import Callback
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class ProgressLogger(Callback):
    """Callback that shows the progress of evaluating metrics.

    :param train_verbose: train verbosity level. 0, 1, or 2, default to 2
        0 = slient, 1 = one line per epoch, 2 = one line per step.
    :type train_verbose: integer
    :param valid_verbose: train verbosity level. 0, 1, or 2, default to 2
        0 = slient, 1 = one line per epoch, 2 = one line per step.
    :type valid_verbose: integer
    :param train_report_steps: report the messages every train steps.
    :type train_report_steps: integer
    :param valid_report_steps: report the messages every valid steps.
    :type valid_report_steps: integer
    """

    def __init__(self):
        """Initialize a ProgressLogger with user-defined verbose levels."""
        super(ProgressLogger, self).__init__()
        self.priority = 270

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.train_verbose = self.trainer.train_verbose
        self.valid_verbose = self.trainer.valid_verbose
        self.train_report_steps = self.trainer.train_report_steps
        self.valid_report_steps = self.trainer.valid_report_steps
        if self.train_report_steps is None:
            self.train_verbose = 0
        if self.valid_report_steps is None:
            self.valid_verbose = 0
        self.total_time_pre_reports = 0
        self.total_time = 0
        logging.debug("Start the unified trainer ... ")
        self.is_chief = self.params['is_chief']
        self.do_validation = self.params['do_validation']

    def before_train_step(self, batch_index, logs=None):
        """Be called before a batch training."""
        self.step_start_time = time.perf_counter()

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoch."""
        self.cur_epoch = epoch
        self.train_num_batches = logs['train_num_batches']
        if self.do_validation:
            self.valid_num_batches = logs['valid_num_batches']

    def after_train_step(self, batch_index, logs=None):
        """Be called before each batch training."""
        self.total_time_pre_reports += time.perf_counter() - self.step_start_time
        if self.train_verbose >= 2 and self.is_chief \
                and batch_index % self.train_report_steps == 0:
            metrics_results = logs.get('train_step_metrics', None)
            lr = logs['lr']
            try:
                cur_loss = logs['cur_loss']
                loss_avg = logs['loss_avg']
            except Exception:
                cur_loss = 0
                loss_avg = 0
                logging.warning("Cant't get the loss, maybe the loss doesn't update in the metric evaluator.")

            current_time = self.total_time_pre_reports / self.train_report_steps
            mean_time = 0
            not_perf_batch = 5
            if batch_index // self.train_report_steps > not_perf_batch:
                self.total_time += self.total_time_pre_reports
                mean_time = self.total_time / (batch_index - not_perf_batch * self.train_report_steps)
            self.total_time_pre_reports = 0
            if metrics_results is not None:
                log_info = "worker id [{}], epoch [{}/{}], train step {}, loss [{:8.3f}, {:8.3f}], " \
                           "lr [{:12.7f}, time [{:4.3f}], mean time [{:4.3f}s], train metrics {}"
                log_info = log_info.format(
                    self.trainer.worker_id,
                    self.cur_epoch + 1, self.trainer.epochs,
                    self._format_batch(batch_index, self.train_num_batches),
                    cur_loss, loss_avg, lr, current_time, mean_time,
                    self._format_metrics(metrics_results))
                logging.info(log_info)
            else:
                log_info = "worker id [{}], epoch [{}/{}], train step {}, loss [{:8.3f}, {:8.3f}], lr [{:12.7f}]" \
                           ", time [{:4.3f}s] , mean time [{:4.3f}s]"
                log_info = log_info.format(
                    self.trainer.worker_id,
                    self.cur_epoch + 1,
                    self.trainer.epochs,
                    self._format_batch(batch_index, self.train_num_batches),
                    cur_loss, loss_avg, lr, current_time, mean_time)
                logging.info(log_info)

    def after_valid_step(self, batch_index, logs=None):
        """Be called after each batch of the validation."""
        if self.valid_verbose >= 2 and self.is_chief \
                and self.do_validation and batch_index % self.valid_report_steps == 0:
            metrics_results = logs.get('valid_step_metrics', None)
            if metrics_results is not None:
                log_info = "worker id [{}], epoch [{}/{}], valid step {},  valid metrics {}".format(
                    self.trainer.worker_id,
                    self.cur_epoch + 1,
                    self.trainer.epochs,
                    self._format_batch(batch_index, self.valid_num_batches),
                    self._format_metrics(metrics_results))
                logging.info(log_info)

    def after_valid(self, logs=None):
        """Be called after validation."""
        if (self.valid_verbose >= 1 and self.is_chief and self.do_validation):
            cur_valid_perfs = logs.get('cur_valid_perfs', None)
            best_valid_perfs = logs.get('best_valid_perfs', None)
            if cur_valid_perfs is not None:
                log_info = "worker id [{}], epoch [{}/{}], current valid perfs {}, best valid perfs {}".format(
                    self.trainer.worker_id,
                    self.cur_epoch + 1,
                    self.trainer.epochs,
                    self._format_metrics(cur_valid_perfs),
                    self._format_metrics(best_valid_perfs))
                logging.info(log_info)

    def after_train(self, logs=None):
        """Be called after the training process."""
        logging.info("Finished the unified trainer successfully.")

    def _format_metrics(self, metrics_results):
        perf_dict = {}
        for name, vals in metrics_results.items():
            if isinstance(vals, np.ndarray):
                # TODO: need a better way to print ndarray
                fmt_vals = "ndarray"
            elif isinstance(vals, Iterable):
                fmt_vals = ','.join(['{:.3f}'.format(item) for item in vals])
            elif vals is None:
                fmt_vals = 'None'
            else:
                fmt_vals = '{:.3f}'.format(vals)
            perf_dict[name] = fmt_vals
        return "[{}]".format(", ".join([": ".join([key, value]) for key, value in perf_dict.items()]))

    def _format_batch(self, batch_index, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        fmt = '[' + fmt + '/' + fmt.format(num_batches) + ']'
        return fmt.format(batch_index)
