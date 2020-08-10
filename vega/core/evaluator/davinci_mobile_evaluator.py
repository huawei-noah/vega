# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""GpuEvaluator used to do evaluate process on gpu."""
import os
import time
import logging
import errno
import pickle
import torch
import numpy as np
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.evaluator.tools.evaluate_davinci_bolt import evaluate
from .conf import DavinciMobileEvaluatorConfig
from vega.core.report import Report, ReportRecord
from .evaluator import Evaluator
from vega.core.trainer.utils import WorkerTypes
from vega.core.metrics.pytorch import Metrics


@ClassFactory.register(ClassType.DAVINCI_MOBILE_EVALUATOR)
class DavinciMobileEvaluator(Evaluator):
    """Evaluator is a davinci and mobile evaluator.

    :param args: arguments from user and default config file
    :type args: dict or Config, default to None
    :param train_data: training dataset
    :type train_data: torch dataset, default to None
    :param valid_data: validate dataset
    :type valid_data: torch dataset, default to None
    :param worker_info: the dict worker info of workers that finished train.
    :type worker_info: dict or None.
    """

    config = DavinciMobileEvaluatorConfig()

    def __init__(self, worker_info=None, model=None, _flag_load_checkpoint=True, **kwargs):
        """Init DavinciMobileEvaluator."""
        super(Evaluator, self).__init__()
        self.framework = self.config.framework
        self.backend = self.config.backend
        self.remote_host = self.config.remote_host
        self.model = model
        self.worker_info = worker_info
        self.worker_type = WorkerTypes.DavinciMobileEvaluator
        if worker_info is not None and "step_name" in worker_info and "worker_id" in worker_info:
            self.step_name = self.worker_info["step_name"]
            self.worker_id = self.worker_info["worker_id"]
        self.checkpoint_file_name = 'checkpoint.pth'
        self.model_pickle_file_name = 'model.pkl'
        if _flag_load_checkpoint:
            self.load_checkpoint()
        else:
            self._load_pretrained_model()
        self.valid_loader = self._init_dataloader(mode='test')

    def valid(self):
        """Validate the latency in davinci or bolt."""
        test_data = "./input.bin"
        latency_sum = 0
        metrics = Metrics(self.config.metric)
        data_num = 0
        for step, batch in enumerate(self.valid_loader):
            if isinstance(batch, list):
                data = batch[0]
                target = batch[1]
            elif isinstance(batch, dict):
                data = batch["LR"] / 255.0
                target = batch["HR"] / 255.0
            else:
                raise ValueError("The dataset formart is invalid.")
            input_shape = data.shape
            data_num += data.size(0)
            if data.size(0) != 1:
                logging.error("The batch_size should be 1, but get {}.".formart(data.size(0)))
            if torch.is_tensor(data):
                data = data.numpy()
            data.tofile(test_data)
            results = evaluate(self.framework, self.backend, self.remote_host,
                               self.model, None, test_data, input_shape)
            latency = np.float(results.get("latency"))
            latency_sum += latency
            output = results.get("out_data")
            output = torch.Tensor(output)
            metrics(output, target)
            if step % self.config.report_freq == 0:
                logging.info("step [{}/{}], latency [{}], valid metric [{}]".format(
                    step + 1, len(self.valid_loader), latency, str(metrics.results)))
        latency_avg = latency_sum / data_num
        logging.info("The latency in {} is {} ms.".format(self.backend, latency_avg))
        pfms = metrics.results
        if self.config.evaluate_latency:
            pfms["latency"] = latency_avg
        logging.info("valid performance: {}".format(pfms))
        return pfms

    def train_process(self):
        """Validate process for the model validate worker."""
        logging.info("start davinci or mobile evaluate process")
        performance = self.valid()
        self._broadcast(performance)
        logging.info("finished davinci or mobile evaluate for id {}".format(self.worker_id))

    def _broadcast(self, pfms):
        """Boadcase pfrm to record."""
        record = Report().receive(self.step_name, self.worker_id)
        if record.performance:
            record.performance.update(pfms)
        else:
            record.performance = pfms
        Report().broadcast(record)
        logging.info("valid record: {}".format(record))
