# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""HostEvaluator used to do evaluate process on gpu."""

import logging
import numpy as np
import vega
from vega.common import ClassFactory, ClassType
from vega.common.general import General
from vega.common.utils import init_log
from .tools.evaluate_davinci_bolt import evaluate
from .conf import DeviceEvaluatorConfig
from vega.report import ReportClient
from .evaluator import Evaluator
from vega.trainer.utils import WorkerTypes
import os
import datetime


@ClassFactory.register(ClassType.DEVICE_EVALUATOR)
class DeviceEvaluator(Evaluator):
    """Evaluator is a Davinci and mobile evaluator.

    :param args: arguments from user and default config file
    :type args: dict or Config, default to None
    :param train_data: training dataset
    :type train_data: torch dataset, default to None
    :param valid_data: validate dataset
    :type valid_data: torch dataset, default to None
    :param worker_info: the dict worker info of workers that finished train.
    :type worker_info: dict or None.
    """

    def __init__(self, worker_info=None, model=None, saved_folder=None, saved_step_name=None,
                 model_desc=None, weights_file=None, **kwargs):
        """Init DeviceEvaluator."""
        super(Evaluator, self).__init__()
        self.config = DeviceEvaluatorConfig()
        # self.backend = self.config.backend
        self.hardware = self.config.hardware
        self.remote_host = self.config.remote_host
        self.intermediate_format = self.config.intermediate_format
        self.calculate_metric = self.config.calculate_metric
        self.quantize = self.config.quantize
        self.model = model
        self.worker_info = worker_info
        self.worker_type = WorkerTypes.DeviceEvaluator
        if worker_info is not None and "step_name" in worker_info and "worker_id" in worker_info:
            self.step_name = self.worker_info["step_name"]
            self.worker_id = self.worker_info["worker_id"]
        self.model_desc = model_desc

        self.weights_file = weights_file
        self.saved_folder = saved_folder
        self.saved_step_name = saved_step_name

    def valid(self):  # noqa: C901
        """Validate the latency in Davinci or bolt."""
        test_data = os.path.join(self.get_local_worker_path(self.step_name, self.worker_id), "input.bin")
        latency_sum = 0
        data_num = 0
        global_step = 0
        error_threshold = int(len(self.valid_loader) * 0.05)
        error_count = 0
        repeat_times = 1
        now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        job_id = self.step_name + "_" + str(self.worker_id) + "_" + now_time
        logging.info("The job id of evaluate service is {}.".format(job_id))
        if vega.is_torch_backend():
            import torch
            from vega.metrics.pytorch import Metrics
            metrics = Metrics(self.config.metric)
            for step, batch in enumerate(self.valid_loader):
                if isinstance(batch, list) or isinstance(batch, tuple):
                    data = batch[0]
                    target = batch[1]
                else:
                    raise ValueError("The dataset format must be tuple or list,"
                                     "but get {}.".format(type(batch)))
                if not self.calculate_metric:
                    repeat_times = 10
                    data = data[0:1]
                    target = target[0:1]

                if not self.calculate_metric and global_step >= 1:
                    break
                if torch.is_tensor(data):
                    data = data.numpy()
                data.tofile(test_data)
                reuse_model = False if global_step == 0 else True
                results = evaluate(backend="pytorch", hardware=self.hardware, remote_host=self.remote_host,
                                   model=self.model, weight=None, test_data=test_data, input_shape=data.shape,
                                   reuse_model=reuse_model, job_id=job_id, repeat_times=repeat_times,
                                   intermediate_format=self.intermediate_format)
                if results.get("status") != "sucess" and error_count <= error_threshold:
                    error_count += 1
                    break
                latency = np.float(results.get("latency"))
                data_num += data.shape[0]
                latency_sum += latency

                if global_step == 0:
                    real_output = self.model(torch.Tensor(data))
                    real_output = real_output.detach().numpy()

                    if isinstance(real_output, tuple):
                        output_shape = real_output[0].shape
                    else:
                        output_shape = real_output.shape
                if self.calculate_metric:
                    out_data = np.array(results.get("out_data")).astype(np.float32)
                    output = out_data.reshape(output_shape)
                    output = torch.Tensor(output)
                    metrics(output, target)
                    pfms = metrics.results
                else:
                    pfms = {}

                global_step += 1
                if global_step % self.config.report_freq == 0:
                    logging.info("step [{}/{}], latency [{}], valid metric [{}]".format(
                        step + 1, len(self.valid_loader), latency, pfms))

        elif vega.is_tf_backend():
            import tensorflow as tf
            from vega.metrics.tensorflow.metrics import Metrics
            tf.reset_default_graph()
            valid_data = self.valid_loader.input_fn()
            metrics = Metrics(self.config.metric)
            iterator = valid_data.make_one_shot_iterator()
            one_element = iterator.get_next()
            total_metric = {}
            avg_metric = {}
            weight_file = self.get_local_worker_path(self.step_name, self.worker_id)
            for step in range(len(self.valid_loader)):
                with tf.Session() as sess:
                    batch = sess.run(one_element)
                data = batch[0]
                target = batch[1]
                if not self.calculate_metric:
                    repeat_times = 10
                    data = data[0:1]
                    target = target[0:1]
                input_shape = data.shape

                if not self.calculate_metric and global_step >= 1:
                    break
                data.tofile(test_data)

                if global_step == 0:
                    input_tf = tf.placeholder(tf.float32, shape=data.shape, name='input_tf')
                    self.model.training = False
                    output = self.model(input_tf)
                    if isinstance(output, tuple):
                        output_shape = output[0].shape
                    else:
                        output_shape = output.shape

                reuse_model = False if global_step == 0 else True
                results = evaluate(backend="tensorflow", hardware=self.hardware, remote_host=self.remote_host,
                                   model=self.model, weight=weight_file, test_data=test_data, input_shape=data.shape,
                                   reuse_model=reuse_model, job_id=job_id, quantize=self.quantize,
                                   repeat_times=repeat_times)
                if results.get("status") != "sucess" and error_count <= error_threshold:
                    error_count += 1
                    break
                latency = np.float(results.get("latency"))
                data_num += input_shape[0]
                latency_sum += latency

                if self.calculate_metric:
                    out_data = np.array(results.get("out_data")).astype(np.float32)
                    output = out_data.reshape(output_shape)
                    target_tf = tf.placeholder(target.dtype, shape=target.shape, name='target_tf')
                    output_tf = tf.placeholder(output.dtype, shape=output.shape, name='output_tf')
                    metrics_dict = metrics(output_tf, target_tf)
                    with tf.Session() as sess:
                        sess.run(tf.local_variables_initializer())
                        for name, metric in metrics_dict.items():
                            tf_metric, tf_metric_update = metric
                            sess.run(tf_metric_update, feed_dict={output_tf: output, target_tf: target})
                            eval_value = sess.run(tf_metric)
                    if global_step == 0:
                        total_metric[name] = eval_value
                    else:
                        total_metric[name] += eval_value
                    avg_metric[name] = total_metric[name] / (global_step + 1)
                    metrics.update(avg_metric)
                    pfms = metrics.results
                else:
                    pfms = {}

                global_step += 1

                if global_step % self.config.report_freq == 0:
                    logging.info("step [{}/{}], latency [{}], valid metric [{}]".format(
                        step + 1, len(self.valid_loader), latency, pfms))

        elif vega.is_ms_backend():
            import mindspore
            from vega.metrics.mindspore import Metrics
            metrics = Metrics(self.config.metric)
            for step, batch in enumerate(self.valid_loader.create_dict_iterator()):
                data = batch["image"]
                target = batch["label"]
                if not self.calculate_metric:
                    repeat_times = 10
                    data = data[0:1]
                    target = target[0:1]

                if not self.calculate_metric and global_step >= 1:
                    break
                data = data.asnumpy()
                data.tofile(test_data)
                reuse_model = False if global_step == 0 else True
                results = evaluate(backend="mindspore", hardware=self.hardware, remote_host=self.remote_host,
                                   model=self.model, weight=None, test_data=test_data, input_shape=data.shape,
                                   reuse_model=reuse_model, job_id=job_id, repeat_times=repeat_times)
                latency = np.float(results.get("latency"))
                latency_sum += latency
                data_num += data.shape[0]

                if global_step == 0:
                    real_output = self.model(mindspore.Tensor(data))
                    real_output = real_output.asnumpy()
                    if isinstance(real_output, tuple):
                        output_shape = real_output[0].shape
                    else:
                        output_shape = real_output.shape
                if self.calculate_metric:
                    out_data = np.array(results.get("out_data")).astype(np.float32)
                    output = out_data.reshape(output_shape)
                    output = mindspore.Tensor(output)
                    metrics(output, target)
                    pfms = metrics.results
                else:
                    pfms = {}

                global_step += 1
                if global_step % self.config.report_freq == 0:
                    logging.info("step [{}/{}], latency [{}], valid metric [{}]".format(
                        step + 1, len(self.valid_loader), latency, pfms))

        latency_avg = latency_sum / data_num
        logging.info("The latency in {} is {} ms.".format(self.hardware, latency_avg))

        if self.config.evaluate_latency:
            pfms["latency"] = latency_avg
        logging.info("valid performance: {}".format(pfms))
        return pfms

    def train_process(self):
        """Validate process for the model validate worker."""
        init_log(level=General.logger.level,
                 log_file=f"{self.step_name}_device_evaluator_{self.worker_id}.log",
                 log_path=self.local_log_path)
        logging.info("start Davinci or mobile evaluate process")
        self.load_model()
        self.valid_loader = self._init_dataloader(mode='test')
        performance = self.valid()
        ReportClient().update(self.step_name, self.worker_id, performance=performance)
        logging.info(f"finished device evaluation, id: {self.worker_id}, performance: {performance}")
