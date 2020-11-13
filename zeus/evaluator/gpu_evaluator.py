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
import zeus
from zeus.common import ClassFactory, ClassType
from zeus.common import init_log
from zeus.common.general import General
from zeus.report import Report
from .conf import GPUEvaluatorConfig
from .evaluator import Evaluator
from zeus.trainer.utils import WorkerTypes


if zeus.is_torch_backend():
    import torch
    from zeus.metrics.pytorch import Metrics
elif zeus.is_tf_backend():
    import tensorflow as tf
    from zeus.metrics.tensorflow.metrics import Metrics
elif zeus.is_ms_backend():
    from zeus.metrics.mindspore.metrics import Metrics
    from mindspore.train import Model as MsModel


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

    def __init__(self, worker_info=None, model=None, saved_folder=None, saved_step_name=None,
                 model_desc=None, weights_file=None, **kwargs):
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
        self.weights_file = weights_file
        self.saved_folder = saved_folder
        self.saved_step_name = saved_step_name

    def valid(self, valid_loader):
        """Validate one step of mode.

        :param loader: valid data loader
        """
        metrics = Metrics(self.config.metric)
        if zeus.is_torch_backend():
            self.model.eval()
            data_num = 0
            latency_sum = 0.0
            with torch.no_grad():
                for step, batch in enumerate(valid_loader):
                    if isinstance(batch, list) or isinstance(batch, tuple):
                        data = batch[0]
                        target = batch[1]
                    else:
                        raise ValueError("The dataset format must be tuple or list,"
                                         "but get {}.".format(type(batch)))
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
        elif zeus.is_tf_backend():
            estimator = self._init_tf_estimator()
            time_start = time.time()
            eval_metrics = estimator.evaluate(input_fn=valid_loader.input_fn, steps=len(valid_loader))
            latency = (time.time() - time_start) / (len(valid_loader) * valid_loader.args.batch_size)
            metrics.update(eval_metrics)
        elif zeus.is_ms_backend():
            metric_name = self.config.metric().type
            dataset_sink_mode = True if zeus.is_npu_device() else False
            ms_model = MsModel(network=self.model,
                               loss_fn=lambda x, y: 0,
                               metrics={metric_name: metrics()})
            time_start = time.time()
            eval_metrics = ms_model.eval(valid_dataset=valid_loader,
                                         callbacks=None,
                                         dataset_sink_mode=dataset_sink_mode)
            for batch in valid_loader.create_dict_iterator():
                batch_size = batch["image"].shape[0]
                break
            latency = (time.time() - time_start) / (valid_loader.get_dataset_size() * batch_size)
            metrics.update(eval_metrics)
        pfms = metrics.results
        if self.config.evaluate_latency:
            pfms["latency"] = latency
        logging.info("evaluate performance: {}".format(pfms))
        return pfms

    def _model_fn(self, features, labels, mode):
        """Model function of gpu evaluator."""
        self.model.training = mode == tf.estimator.ModeKeys.TRAIN
        logits = self.model(features)
        logits = tf.cast(logits, tf.float32)
        eval_metric_ops = Metrics(self.config.metric)(logits, labels)
        loss = tf.losses.absolute_difference(features, features)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=None,
                                          eval_metric_ops=eval_metric_ops)

    def _init_tf_estimator(self):
        """Estimator of gpu evaluator used in tf backend."""
        config = tf.estimator.RunConfig(model_dir=self.saved_folder,
                                        log_step_count_steps=self.config.report_freq,
                                        session_config=tf.compat.v1.ConfigProto())
        return tf.estimator.Estimator(model_fn=self._model_fn, config=config)

    def train_process(self):
        """Validate process for the model validate worker."""
        init_log(level=General.logger.level,
                 log_file="log_gpu_eva_{}.txt".format(self.worker_id),
                 log_path=self.local_log_path)
        logging.info("start evaluate process")
        self.load_model()
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
        logging.debug("evaluate record: {}".format(record))
