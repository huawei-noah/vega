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

"""DetectionProgressLogger call defination."""

import logging
import time
from collections import OrderedDict
from prettytable import PrettyTable
from vega.common import ClassFactory, ClassType
from vega.trainer.callbacks.progress_logger import ProgressLogger


@ClassFactory.register(ClassType.CALLBACK)
class DetectionProgressLogger(ProgressLogger):
    """Callback that shows the progress of evaluating metrics."""

    def after_train_step(self, batch_index, logs=None):
        """Be called before each batch training."""
        if self.train_verbose >= 2 and self.trainer.is_chief \
                and batch_index % self.train_report_steps == 0:
            try:
                out_buffer = OrderedDict(
                    time=time.strftime("%Y-%m-%d @ %H:%M:%S"),
                    epoch=f'{self.cur_epoch}/{self.trainer.epochs}',
                    step=f'{self._format_batch(batch_index, self.train_num_batches)}',
                    lr=f'{self.trainer.optimizer.param_groups[0]["lr"]:.8f}',
                    cls_pos_loss=f'{logs["cls_pos_loss"]:.3f}',
                    cls_neg_loss=f'{logs["cls_neg_loss"]:.3f}',
                    loc_loss=f'{logs["loc_loss"]:.3f}',
                    cur_loss=f'{logs["cur_loss"]:.3f}',
                    loss_avg=f'{logs["loss_avg"]:.3f}',
                )
                pt = PrettyTable()
                pt.field_names = out_buffer.keys()
                pt.add_row(out_buffer.values())
            except Exception:
                logging.warning("Cant't get the loss, maybe the loss doesn't update in the metric evaluator.")
            logging.info('\n' + pt.get_string())

    def after_valid_step(self, batch_index, logs=None):
        """Be called after each batch of the validation."""
        if self.valid_verbose >= 2 and self.trainer.is_chief \
                and self.trainer.do_validation and batch_index % self.valid_report_steps == 0:
            metrics_results = logs.get('valid_step_metrics', None)
            if metrics_results is not None:
                out_buffer = OrderedDict(
                    time=time.strftime("%Y-%m-%d @ %H:%M:%S"),
                    epoch=f'{self.cur_epoch}/{self.trainer.epochs}',
                    step=f'{self._format_batch(batch_index, self.valid_num_batches)}',
                )
                out_buffer.update(metrics_results)
                pt = PrettyTable()
                pt.field_names = out_buffer.keys()
                pt.add_row(out_buffer.values())

    def after_valid(self, logs=None):
        """Be called after validation."""
        if (self.valid_verbose >= 1 and self.trainer.is_chief and self.trainer.do_validation):
            cur_valid_perfs = logs.get('cur_valid_perfs', None)
            if cur_valid_perfs is not None:
                log_info = "epoch [{}/{}], current valid perfs {}".format(
                    self.cur_epoch + 1, self.trainer.epochs, self._format_metrics(cur_valid_perfs))
                logging.info(log_info)
