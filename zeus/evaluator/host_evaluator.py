# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""HostEvaluator used to do evaluate process on gpu."""
import time
import logging
import zeus
from zeus.common import ClassFactory, ClassType
from zeus.common import init_log
from zeus.common.general import General
from zeus.report import ReportClient
from .conf import HostEvaluatorConfig
from .evaluator import Evaluator
from zeus.trainer.utils import WorkerTypes


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

    def valid(self, valid_loader):
        """Validate one step of mode.

        :param loader: valid data loader
        """
        if zeus.is_torch_backend():
            import torch
            from zeus.metrics.pytorch import Metrics
            metrics = Metrics(self.config.metric)
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
            from zeus.metrics.tensorflow.metrics import Metrics
            metrics = Metrics(self.config.metric)
            estimator = self._init_tf_estimator()
            time_start = time.time()
            eval_metrics = estimator.evaluate(input_fn=valid_loader.input_fn, steps=len(valid_loader))
            latency = (time.time() - time_start) / (len(valid_loader) * valid_loader.args.batch_size)
            metrics.update(eval_metrics)
        elif zeus.is_ms_backend():
            from zeus.metrics.mindspore.metrics import Metrics
            from mindspore.train import Model as MsModel
            from .utils import FakeLoss
            metrics = Metrics(self.config.metric)
            metric_name = self.config.metric().type
            dataset_sink_mode = True if zeus.is_npu_device() else False
            # when eval, the loss_fn is not needed actually, but when initilized, the loss_fn can't be None
            ms_model = MsModel(network=self.model,
                               loss_fn=FakeLoss(),
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
        import tensorflow as tf
        from zeus.metrics.tensorflow.metrics import Metrics
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

    def train_process(self):
        """Validate process for the model validate worker."""
        init_log(level=General.logger.level,
                 log_file="host_evaluator_{}.log".format(self.worker_id),
                 log_path=self.local_log_path)
        logging.info("start evaluate process")
        self.load_model()
        self.valid_loader = self._init_dataloader(mode='test')
        performance = self.valid(self.valid_loader)
        self._broadcast(performance)
        logging.info("the model (id {}) is evaluated on the host".format(self.worker_id))

    def _broadcast(self, pfms):
        """Boadcase pfrm to record."""
        record = ReportClient.get_record(self.step_name, self.worker_id)
        if record.performance:
            record.performance.update(pfms)
        else:
            record.performance = pfms
        ReportClient.broadcast(record)
        logging.debug("evaluate record: {}".format(record))

    def _init_session_config(self):
        import tensorflow as tf
        if zeus.is_gpu_device():
            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            return sess_config
        elif zeus.is_npu_device():
            from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
            sess_config = tf.ConfigProto()
            sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
            custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name = "NpuOptimizer"
            custom_op.parameter_map["use_off_line"].b = True
            return sess_config
