# -*- coding: utf-8 -*-

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

"""HostEvaluator used to do evaluate process on gpu."""

import datetime
import logging
import os
import traceback
import numpy as np
import vega
from vega.common import ClassFactory, ClassType
from vega.common.wrappers import train_process_wrapper
from vega.report import ReportClient
from vega.trainer.utils import WorkerTypes
from .conf import DeviceEvaluatorConfig
from .evaluator import Evaluator
from .tools.evaluate_davinci_bolt import evaluate


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
        self.hardware = self.config.hardware
        self.remote_host = self.config.remote_host
        self.intermediate_format = self.config.intermediate_format
        self.opset_version = self.config.opset_version
        self.precision = self.config.precision.upper()
        self.calculate_metric = self.config.calculate_metric
        self.muti_input = self.config.muti_input
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
        self.custom = self.config.custom
        self.repeat_times = self.config.repeat_times

    def _get_data(self):
        def _get_data_default():
            if vega.is_torch_backend():
                for batch in self.valid_loader:
                    data = batch[0]
                    break
            else:
                for batch in self.valid_loader.create_dict_iterator():
                    data = batch["image"]
                    data = data.asnumpy()
                    break
            return data

        if self.custom is None:
            data = _get_data_default()
            reshape_batch_size = self.config.reshape_batch_size
            if reshape_batch_size and isinstance(reshape_batch_size, int):
                data = data[0:reshape_batch_size]
        else:
            custom_cls = ClassFactory.get_cls(ClassType.DEVICE_EVALUATOR, self.custom)(self)
            self.custom = custom_cls
            data = self.custom.get_data()

        return data

    def valid(self):
        """Validate the latency in Davinci or bolt."""
        test_data = os.path.join(self.get_local_worker_path(self.step_name, self.worker_id), "input.bin")
        latency_sum = 0
        data_num = 0
        global_step = 0
        now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        job_id = self.step_name + "_" + str(self.worker_id) + "_" + now_time
        logging.info("The job id of evaluate service is {}.".format(job_id))
        if vega.is_torch_backend():
            pfms, latency_sum, data_num = self._torch_valid(test_data, job_id)
        elif vega.is_tf_backend():
            pfms, latency_sum, data_num = self._tf_valid(test_data, latency_sum, data_num, global_step, job_id)
        elif vega.is_ms_backend():
            pfms, latency_sum, data_num = self._ms_valid(test_data, job_id)
        latency_avg = latency_sum / data_num
        logging.info("The latency in {} is {} ms.".format(self.hardware, latency_avg))

        if self.config.evaluate_latency:
            pfms["latency"] = latency_avg
        logging.info("valid performance: {}".format(pfms))
        return pfms

    def _torch_valid(self, test_data, job_id):
        import torch
        if self.calculate_metric:
            return self._torch_valid_metric(test_data, job_id)

        data = self._get_data()
        if torch.is_tensor(data):
            data = data.numpy()
        data.tofile(test_data)
        results = evaluate(backend="pytorch", hardware=self.hardware, remote_host=self.remote_host,
                           model=self.model, weight=None, test_data=test_data, input_shape=data.shape,
                           reuse_model=False, job_id=job_id, precision=self.precision,
                           cal_metric=self.calculate_metric,
                           intermediate_format=self.intermediate_format,
                           opset_version=self.opset_version, repeat_times=self.repeat_times,
                           save_intermediate_file=self.config.save_intermediate_file, custom=self.custom,
                           muti_input=self.muti_input)

        latency = np.float(results.get("latency"))
        data_num = 1
        pfms = {}
        return pfms, latency, data_num

    def _torch_valid_metric(self, test_data, job_id):
        import torch
        from vega.metrics.pytorch import Metrics
        metrics = Metrics(self.config.metric)
        latency_sum = 0
        error_count = 0
        error_threshold = int(len(self.valid_loader) * 0.05)
        for step, batch in enumerate(self.valid_loader):
            if isinstance(batch, list) or isinstance(batch, tuple):
                data = batch[0]
                target = batch[1]
            else:
                raise ValueError("The dataset format must be tuple or list,"
                                 "but get {}.".format(type(batch)))
            if torch.is_tensor(data):
                data = data.numpy()
            data.tofile(test_data)
            reuse_model = False if step == 0 else True
            results = evaluate(backend="pytorch", hardware=self.hardware, remote_host=self.remote_host,
                               model=self.model, weight=None, test_data=test_data, input_shape=data.shape,
                               reuse_model=reuse_model, job_id=job_id,
                               precision=self.precision, cal_metric=self.calculate_metric,
                               intermediate_format=self.intermediate_format,
                               opset_version=self.opset_version, repeat_times=self.repeat_times,
                               save_intermediate_file=self.config.save_intermediate_file, muti_input=self.muti_input)
            if results.get("status") != "success" and error_count <= error_threshold:
                error_count += 1
                break
            latency = np.float(results.get("latency"))
            latency_sum += latency

            if step == 0:
                self.model.eval()
                real_output = self.model(torch.Tensor(data))
                real_output = real_output.detach().numpy()

                if isinstance(real_output, tuple):
                    output_shape = real_output[0].shape
                else:
                    output_shape = real_output.shape
            out_data = np.array(results.get("out_data")).astype(np.float32)
            output = out_data.reshape(output_shape)
            output = torch.Tensor(output)
            metrics(output, target)
            pfms = metrics.results

            if step % self.config.report_freq == 0:
                logging.info("step [{}/{}], latency [{}], valid metric [{}]".format(
                    step + 1, len(self.valid_loader), latency, pfms))

        return pfms, latency_sum, step

    def _tf_valid(self, test_data, latency_sum, data_num, global_step, job_id):
        import tensorflow as tf
        from vega.metrics.tensorflow.metrics import Metrics
        error_count = 0
        error_threshold = int(len(self.valid_loader) * 0.05)
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
                reshape_batch_size = self.config.reshape_batch_size
                if reshape_batch_size and isinstance(reshape_batch_size, int):
                    data = data[0:reshape_batch_size]
                    target = target[0:reshape_batch_size]

            if not self.calculate_metric and global_step >= 1:
                break
            data.tofile(test_data)

            if global_step == 0 and self.calculate_metric:
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
                               repeat_times=self.repeat_times, precision=self.precision,
                               cal_metric=self.calculate_metric,
                               save_intermediate_file=self.config.save_intermediate_file, muti_input=self.muti_input)
            if self.calculate_metric and results.get("status") != "success" and error_count <= error_threshold:
                error_count += 1
                break
            latency = np.float(results.get("latency"))
            data_num += 1
            latency_sum += latency

            if self.calculate_metric:
                pfms = self._calc_tf_metric(
                    results, output_shape, target, metrics, global_step, total_metric, avg_metric)
            else:
                pfms = {}

            global_step += 1

            if global_step % self.config.report_freq == 0:
                logging.info("step [{}/{}], latency [{}], valid metric [{}]".format(
                    step + 1, len(self.valid_loader), latency, pfms))
        return pfms, latency_sum, data_num

    def _calc_tf_metric(self, results, output_shape, target, metrics, global_step, total_metric, avg_metric):
        import tensorflow as tf
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
        return metrics.results

    def _ms_valid(self, test_data, job_id):
        if self.calculate_metric:
            return self._ms_valid_metric(test_data, job_id)

        data = self._get_data()
        data.tofile(test_data)
        results = evaluate(
            backend="mindspore", hardware=self.hardware, remote_host=self.remote_host,
            model=self.model, weight=None, test_data=test_data, input_shape=data.shape,
            reuse_model=False, job_id=job_id, precision=self.precision, cal_metric=self.calculate_metric,
            repeat_times=self.repeat_times,
            save_intermediate_file=self.config.save_intermediate_file, custom=self.custom, muti_input=self.muti_input)
        latency = np.float(results.get("latency"))
        pfms = {}
        data_num = 1
        return pfms, latency, data_num

    def _ms_valid_metric(self, test_data, job_id):
        import mindspore
        from vega.metrics.mindspore import Metrics
        metrics = Metrics(self.config.metric)
        latency_sum = 0
        for step, batch in enumerate(self.valid_loader.create_dict_iterator()):
            data = batch["image"]
            target = batch["label"]
            data = data.asnumpy()
            data.tofile(test_data)
            reuse_model = False if step == 0 else True
            results = evaluate(
                backend="mindspore", hardware=self.hardware, remote_host=self.remote_host,
                model=self.model, weight=None, test_data=test_data, input_shape=data.shape,
                reuse_model=reuse_model, job_id=job_id, precision=self.precision, cal_metric=self.calculate_metric,
                repeat_times=self.repeat_times,
                save_intermediate_file=self.config.save_intermediate_file, muti_input=self.muti_input)
            latency = np.float(results.get("latency"))
            latency_sum += latency

            if step == 0:
                real_output = self.model(mindspore.Tensor(data))
                real_output = real_output.asnumpy()
                if isinstance(real_output, tuple):
                    output_shape = real_output[0].shape
                else:
                    output_shape = real_output.shape

            out_data = np.array(results.get("out_data")).astype(np.float32)
            output = out_data.reshape(output_shape)
            output = mindspore.Tensor(output)
            metrics(output, target)
            pfms = metrics.results

            if step % self.config.report_freq == 0:
                logging.info("step [{}/{}], latency [{}], valid metric [{}]".format(
                    step + 1, len(self.valid_loader), latency, pfms))
        return pfms, latency_sum, step

    @train_process_wrapper
    def train_process(self):
        """Validate process for the model validate worker."""
        try:
            self.load_model()
            if self.custom is None:
                self.valid_loader = self._init_dataloader(mode='test')
            performance = self.valid()
            ReportClient().update(self.step_name, self.worker_id, performance=performance)
            logging.info(f"finished device evaluation, id: {self.worker_id}, performance: {performance}")
        except Exception as e:
            logging.debug(traceback.format_exc())
            logging.error(f"Failed to evalute on device, message: {e}.")
