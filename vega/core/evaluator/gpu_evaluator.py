# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""GpuEvaluator used to do evaluate process on gpu."""
import time
import logging
import torch
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import init_log
from vega.core.metrics.pytorch import Metrics
from vega.core.report import Report, ReportRecord
from .conf import GPUEvaluatorConfig
from .evaluator import Evaluator
from vega.core.trainer.utils import WorkerTypes
from vega.model_zoo import ModelZoo


@ClassFactory.register(ClassType.GPU_EVALUATOR)
class GpuEvaluator(Evaluator):
    """Evaluator is a gpu evaluator.

    :param args: arguments from user and default config file
    :type args: dict or Config, default to None
    :param train_data: training dataset
    :type train_data: torch dataset, default to None
    :param valid_data: validate dataset
    :type valid_data: torch dataset, default to None
    :param worker_info: the dict worker info of workers that finished train.
    :type worker_info: dict or None.

    """

    config = GPUEvaluatorConfig()

    def __init__(self, worker_info=None, model=None, _flag_load_checkpoint=True,
                 saved_folder=None, saved_step_name=None, model_desc=None, weights_file=None, **kwargs):
        """Init GpuEvaluator."""
        super(Evaluator, self).__init__()
        self.worker_info = worker_info
        self.worker_type = WorkerTypes.GPU_EVALUATOR
        if worker_info is not None and "step_name" in worker_info and "worker_id" in worker_info:
            self.step_name = self.worker_info["step_name"]
            self.worker_id = self.worker_info["worker_id"]
        self.model = model
        self.model_desc = model_desc
        self.evaluate_result = None
        self.checkpoint_file_name = 'checkpoint.pth'
        self.model_pickle_file_name = 'model.pkl'
        self.weights_file = weights_file
        self._flag_load_checkpoint = _flag_load_checkpoint
        self.saved_folder = saved_folder
        self.saved_step_name = saved_step_name

    def valid(self, valid_loader):
        """Validate one step of mode.

        :param loader: valid data loader
        """
        self.model.eval()
        metrics = Metrics(self.config.metric)
        data_num = 0
        latency_sum = 0.0
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                if isinstance(batch, list):
                    data = batch[0]
                    target = batch[1]
                elif isinstance(batch, dict):
                    data = batch["LR"] / 255.0
                    target = batch["HR"] / 255.0
                else:
                    raise ValueError("The dataset formart is invalid.")
                if self.config.cuda:
                    data, target = data.cuda(), target.cuda()
                    self.model = self.model.cuda()
                time_start = time.time()
                logits = self.model(data)
                latency_sum += time.time() - time_start
                metrics(logits, target)
                n = data.size(0)
                data_num += n
                if step % self.config.report_freq == 0:
                    logging.info("step [{}/{}], valid metric [{}]".format(
                        step + 1, len(valid_loader), str(metrics.results)))
        latency = latency_sum / data_num
        pfms = metrics.results
        if self.config.evaluate_latency:
            pfms["latency"] = latency
        logging.info("valid performance: {}".format(pfms))
        return pfms

    def train_process(self):
        """Validate process for the model validate worker."""
        init_log(log_file="gpu_eva_{}.txt".format(self.worker_id))
        logging.info("start evaluate process")
        if self.model_desc and self.weights_file:
            self.model = ModelZoo.get_model(self.model_desc, self.weights_file)
        elif self._flag_load_checkpoint:
            self.load_checkpoint(saved_folder=self.saved_folder, step_name=self.saved_step_name)
        else:
            self._load_pretrained_model()
        self.valid_loader = self._init_dataloader(mode='test')
        performance = self.valid(self.valid_loader)
        self._broadcast(performance)
        logging.info("finished evaluate for id {}".format(self.worker_id))

    def _broadcast(self, pfms):
        """Boadcase pfrm to record."""
        record = Report().receive(self.step_name, self.worker_id)
        if record.performance:
            record.performance.update(pfms)
        else:
            record.performance = pfms
        Report().broadcast(record)
        logging.debug("valid record: {}".format(record))
