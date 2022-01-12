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

"""Custom callbacks used in mindspore."""

import logging
from mindspore.train.callback import Callback
from vega.report import ReportClient


class EvalCallBack(Callback):
    """
    Monitor the metric in training.

    :param model: the mindspore model
    :type model: Class of mindspore.train.Model
    :param eval_dataset: valid dataloader
    :type eval_dataset: MindDataset
    """

    def __init__(self, model, eval_dataset, dataset_sink_mode, trainer):
        super(EvalCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.dataset_sink_mode = dataset_sink_mode
        self.trainer = trainer

    def epoch_end(self, run_context):
        """Be called after each epoch."""
        cb_params = run_context.original_args()
        metric = self.model.eval(self.eval_dataset, dataset_sink_mode=self.dataset_sink_mode)
        logging.info("Current epoch : [{}/{}], current valid metric {}.".format(
            cb_params.cur_epoch_num, cb_params.epoch_num, metric))

        self.trainer.performance.update(metric)
        if not self.trainer.is_chief:
            return
        else:
            ReportClient().update(
                self.trainer.step_name,
                self.trainer.worker_id,
                num_epochs=cb_params.epoch_num,
                current_epoch=cb_params.cur_epoch_num,
                performance=self.trainer.performance)
