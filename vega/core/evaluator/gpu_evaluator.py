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
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.trainer.pytorch.trainer import Trainer
from vega.core.trainer.utils import WorkerTypes
from vega.core.common import FileOps, init_log
from vega.datasets.pytorch import Dataset
from vega.core.metrics.pytorch import Metrics
from vega.core.common.utils import update_dict


@ClassFactory.register(ClassType.GPU_EVALUATOR)
class GpuEvaluator(Trainer):
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

    def __init__(self, worker_info=None, model=None, hps=None, load_checkpoint=False, **kwargs):
        """Init GpuEvaluator."""
        self._reference_trainer_settings()
        super(GpuEvaluator, self).__init__(self.cfg)
        self.worker_type = WorkerTypes.GPU_EVALUATOR
        self.worker_info = worker_info
        if worker_info is not None and "step_name" in worker_info and "worker_id" in worker_info:
            self.step_name = self.worker_info["step_name"]
            self.worker_id = self.worker_info["worker_id"]
        self._flag_load_checkpoint = load_checkpoint
        self.hps = hps
        self.model = model
        self.evaluate_result = None

    def _reference_trainer_settings(self):
        """Set reference Trainer."""
        ref = self.cfg.get('ref')
        if ref:
            ref_dict = ClassFactory.__configs__
            for key in ref.split('.'):
                ref_dict = ref_dict.get(key)
            update_dict(ref_dict, self.cfg)

    def _init_all_settings(self):
        """Init all settings from config."""
        self._reference_trainer_settings()
        if self.cfg.cuda:
            self._init_cuda_setting()
        self._init_hps(self.hps)
        if self.model is None:
            self.model = self._init_model()
        if self.model is not None and self.cfg.cuda:
            self.model = self.model.cuda()
        # TODO
        if self._flag_load_checkpoint:
            self.load_checkpoint()
        else:
            self._load_pretrained_model()
        self._init_dataloader()

    def _init_dataloader(self):
        """Init dataloader."""
        valid_dataset = Dataset(mode='test')
        self.valid_loader = valid_dataset.dataloader

    def valid(self, valid_loader):
        """Validate one step of mode.

        :param loader: valid data loader
        """
        self.model.eval()
        metrics = Metrics(self.cfg.metric)
        data_num = 0
        latency_sum = 0.0
        with torch.no_grad():
            for step, (data, target) in enumerate(valid_loader):
                if self.cfg.cuda:
                    data, target = data.cuda(), target.cuda()
                    self.model = self.model.cuda()
                time_start = time.time()
                logits = self.model(data)
                latency_sum += time.time() - time_start
                metrics(logits, target)
                n = data.size(0)
                data_num += n
                if self._first_rank and step % self.cfg.report_freq == 0:
                    logging.info("step [{}/{}], valid metric [{}]".format(
                        step + 1, len(valid_loader), str(metrics.results_dict)))
        latency = latency_sum / data_num
        pfms = metrics.results_dict
        performance = [pfms[list(pfms.keys())[0]]]
        if self.cfg.evaluate_latency:
            performance.append(latency)
        logging.info("valid performance: {}".format(performance))
        return performance

    def train_process(self):
        """Validate process for the model validate worker."""
        init_log(log_file="gpu_eva_{}.txt".format(self.worker_id))
        logging.info("start evaluate process")
        self._init_all_settings()
        performance = self.valid(self.valid_loader)
        self._save_performance(performance)
        logging.info("finished evaluate for id {}".format(self.worker_id))
        self.evaluate_result = performance
        return
