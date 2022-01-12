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

import time
import copy
import logging
from statistics import mean
import traceback
import vega
from vega.common import ClassFactory, ClassType
from vega.common.general import General
from vega.common.wrappers import train_process_wrapper
from vega.report import ReportClient
from vega.trainer.utils import WorkerTypes
from .conf import HostEvaluatorConfig
from .evaluator import Evaluator


@ClassFactory.register(ClassType.HOST_EVALUATOR)
class HostEvaluator(Evaluator):
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

    def __init__(self, worker_info=None, model=None, saved_folder=None, saved_step_name=None,
                 model_desc=None, weights_file=None, **kwargs):
        """Init HostEvaluator."""
        super(Evaluator, self).__init__()
        self.config = HostEvaluatorConfig()
        self.worker_info = worker_info
        self.worker_type = WorkerTypes.HOST_EVALUATOR
        if worker_info is not None and "step_name" in worker_info and "worker_id" in worker_info:
            self.step_name = self.worker_info["step_name"]
            self.worker_id = self.worker_info["worker_id"]
        self.model = model
        self.model_desc = model_desc
        self.evaluate_result = None
        self.weights_file = weights_file
        self.saved_folder = saved_folder
        self.saved_step_name = saved_step_name

    def _call_model_batch(self, batch):
        input, target = None, None
        if isinstance(batch, dict):
            logits = self.model(**batch)
        elif isinstance(batch, list) and isinstance(batch[0], dict):
            target = batch
            logits = self.model(batch)
        else:
            input, target = batch
            logits = self.model(input) if not isinstance(input, dict) else self.model(**input)
        return logits, target

    def valid(self, valid_loader):
        """Validate one step of mode.

        :param loader: valid data loader
        """
        if vega.is_torch_backend():
            import torch
            from vega.metrics.pytorch import Metrics
            if vega.is_gpu_device():
                self.model = self.model.cuda()
            elif vega.is_npu_device():
                self.model = self.model.to(vega.get_devices())
            metrics = Metrics(self.config.metric)
            self.model.eval()
            latency_batch = None
            cal_lantency_counts = 10
            with torch.no_grad():
                for step, batch in enumerate(valid_loader):
                    batch = self._set_device(batch)
                    if not latency_batch:
                        latency_batch = copy.deepcopy(batch)
                    logits, target = self._call_model_batch(batch)
                    metrics_results = metrics(logits, target)
                    if step % self.config.report_freq == 0 and metrics_results:
                        logging.info(
                            "step [{}/{}], valid metric [{}]".format(step + 1, len(valid_loader), metrics_results))
                latency_pre_batch = []
                for i in range(cal_lantency_counts):
                    time_init = time.perf_counter()
                    self._call_model_batch(latency_batch)
                    latency_pre_batch.append((time.perf_counter() - time_init) * 1000)
                latency = mean(latency_pre_batch)
            logging.info("evaluator latency [{}]".format(latency))
        elif vega.is_tf_backend():
            from vega.metrics.tensorflow.metrics import Metrics
            metrics = Metrics(self.config.metric)
            estimator = self._init_tf_estimator()
            time_start = time.time()
            eval_metrics = estimator.evaluate(input_fn=valid_loader.input_fn, steps=len(valid_loader))
            latency = (time.time() - time_start) / (len(valid_loader) * valid_loader.args.batch_size)
            metrics.update(eval_metrics)
        elif vega.is_ms_backend():
            from vega.metrics.mindspore.metrics import Metrics
            from mindspore.train import Model as MsModel
            from .utils import FakeLoss
            self._init_ms_context()
            metrics = Metrics(self.config.metric)
            metric_name = self.config.metric().type
            ms_metric = metrics() if isinstance(metrics(), dict) else {metric_name: metrics()}
            # when eval, the loss_fn is not needed actually, but when initilized, the loss_fn can't be None
            ms_model = MsModel(network=self.model,
                               loss_fn=FakeLoss(),
                               metrics=ms_metric)
            time_start = time.time()
            eval_metrics = ms_model.eval(valid_dataset=valid_loader,
                                         callbacks=None,
                                         dataset_sink_mode=self.dataset_sink_mode)
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

    def _set_device(self, data):
        import torch
        if torch.is_tensor(data):
            if vega.is_gpu_device():
                return data.cuda()
            else:
                return data.to(vega.get_devices())
        if isinstance(data, dict):
            return {k: self._set_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._set_device(v) for v in data]
        elif isinstance(data, tuple):
            return tuple([self._set_device(v) for v in data])
        return data

    def _model_fn(self, features, labels, mode):
        """Model function of gpu evaluator."""
        import tensorflow as tf
        from vega.metrics.tensorflow.metrics import Metrics
        self.model.training = mode == tf.estimator.ModeKeys.TRAIN
        logits = self.model(features)
        logits = tf.cast(logits, tf.float32)
        eval_metric_ops = Metrics(self.config.metric)(logits, labels)
        return tf.estimator.EstimatorSpec(mode=mode, loss=tf.log(1.0), train_op=None, eval_metric_ops=eval_metric_ops)

    def _init_tf_estimator(self):
        """Estimator of gpu evaluator used in tf backend."""
        import tensorflow as tf
        session_config = self._init_session_config()
        config = tf.estimator.RunConfig(model_dir=self.saved_folder,
                                        log_step_count_steps=self.config.report_freq,
                                        session_config=session_config)
        return tf.estimator.Estimator(model_fn=self._model_fn, config=config)

    @train_process_wrapper
    def train_process(self):
        """Validate process for the model validate worker."""
        logging.info("start evaluate process")
        try:
            self.load_model()
            self.valid_loader = self._init_dataloader(mode='test')
            performance = self.valid(self.valid_loader)
            ReportClient().update(self.step_name, self.worker_id, performance=performance)
            logging.info(f"finished host evaluation, id: {self.worker_id}, performance: {performance}")
        except Exception as e:
            logging.debug(traceback.format_exc())
            logging.error(f"Failed to evalute on host, message: {e}")

    def _init_session_config(self):
        import tensorflow as tf
        if vega.is_gpu_device():
            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            return sess_config
        elif vega.is_npu_device():
            from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
            # Initialize npu bridge
            from npu_bridge import npu_init
            sess_config = tf.ConfigProto()
            sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
            custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name = "NpuOptimizer"
            custom_op.parameter_map["use_off_line"].b = True
            return sess_config

    def _init_ms_context(self):
        from mindspore import context
        mode = General.ms_execute_mode
        logging.info(f"Run evaluator in mode: {mode}.")
        if vega.is_npu_device():
            context.set_context(mode=mode, device_target="Ascend")
        else:
            context.set_context(mode=mode, device_target="CPU")
        self.dataset_sink_mode = General.dataset_sink_mode
        logging.info(f"Dataset_sink_mode:{self.dataset_sink_mode}.")
