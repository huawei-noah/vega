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

"""TensorFlow Trainer."""

import logging
import tensorflow as tf
import vega
from vega.trainer.trainer_base import TrainerBase
from vega.modules.loss import Loss
from vega.trainer.modules.lr_schedulers import LrScheduler
from vega.trainer.modules.optimizer import Optimizer
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRAINER)
class TrainerTf(TrainerBase):
    """Trainer tensorflow class."""

    def build(self):
        """Build the trainer by assembling the necessary components."""
        super().build()
        self.train_metrics = None
        self.valid_metrics = self._init_metrics()

    def set_training_settings(self):
        """Set trainer training settings."""
        self.model_fn = self._default_model_fn
        self.train_input_fn = self._default_train_input_fn
        self.valid_input_fn = self._default_valid_input_fn
        self._init_tf_session()
        self._init_distributed_setting()
        self._init_tf_estimator()

    def _train_epoch(self):
        if self.config.train_in_once:
            max_steps = self.config.max_train_steps or len(self.train_loader) * self.epochs
            self.estimator.train(input_fn=self.train_input_fn,
                                 max_steps=max_steps,
                                 hooks=self._init_logging_hook())
        else:
            self.estimator.train(input_fn=self.train_input_fn,
                                 steps=self.config.max_train_steps or len(self.train_loader),
                                 hooks=self._init_logging_hook())

    def _valid_epoch(self):
        self.callbacks.before_valid()
        valid_logs = None

        eval_metrics = self.estimator.evaluate(input_fn=self.valid_input_fn,
                                               steps=len(self.valid_loader))
        self.valid_metrics.update(eval_metrics)
        valid_logs = dict()
        valid_logs['cur_valid_perfs'] = self.valid_metrics.results

        self.callbacks.after_valid(valid_logs)

    def _init_distributed_setting(self):
        if self.hccl:
            sess_config = self._init_session_config()
            self.sess = tf.compat.v1.Session(config=sess_config)
            from npu_bridge.estimator import npu_ops
            self.npu_init = npu_ops.initialize_system()
            self.npu_shutdown = npu_ops.shutdown_system()
            self.sess.run(self.npu_init)

    def _default_train_input_fn(self):
        return self.train_loader.input_fn()

    def _default_valid_input_fn(self):
        return self.valid_loader.input_fn()

    def _default_model_fn(self, features, labels, mode):
        """Define model_fn used by TensorFlow Estimator.

        :params features: input features
        :type features: tensorflow tensors
        :params labels: label data
        :type labels: tensorflow tensors
        :params mode: mode of estimator
        :type mode: tf.estimator.ModeKeys
        :return: tensorflow EstimatorSpec
        :rtype: tf.estimator.EstimatorSpec
        """
        logging.info('model function action')

        self.model.training = mode == tf.estimator.ModeKeys.TRAIN
        if self.config.mixup and mode == tf.estimator.ModeKeys.TRAIN:
            mixup_ratio = tf.compat.v1.distributions.Beta(0.1, 0.1).sample()
            mixed_x, y_a, y_b = self._mixup_batch(features, labels, mixup_ratio)
            logits = self.model(mixed_x)
        else:
            logits = self.model(features)
        logits = tf.cast(logits, tf.float32)
        if hasattr(self.model, 'add_loss'):
            loss_cls = Loss()()
            self.model.add_loss(loss_cls)
            self.loss = self.model.overall_loss()
        else:
            self.loss = Loss()()
        if self.config.mixup and mode == tf.estimator.ModeKeys.TRAIN:
            loss = self._mixup_loss(self.loss, logits, y_a, y_b, mixup_ratio)
        else:
            loss = self.loss(logits, labels)
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.compat.v1.train.get_or_create_global_step()
            epoch = tf.cast(global_step, tf.float32) / tf.cast(len(self.train_loader), tf.float32)
            distributed = self.horovod or self.hccl
            self.optimizer = Optimizer()(distributed=distributed)
            self.lr_scheduler = LrScheduler()(optimizer=self.optimizer)
            self.lr_scheduler.step(epoch)
            if distributed:
                self.optimizer = Optimizer.set_distributed(self.optimizer)

            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            loss_scale = self.config.loss_scale if self.use_amp else 1
            minimize_op = self.optimizer.step(loss, loss_scale, global_step)
            train_op = tf.group(minimize_op, update_ops)
            logging_hook = list()
            logging_hook.append(tf.train.LoggingTensorHook(
                tensors={"learning rate": self.lr_scheduler.get_lr()[0]},
                every_n_iter=self.config.train_report_steps))

        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.valid_metrics(logits, labels)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op,
                eval_metric_ops=eval_metric_ops,
                training_hooks=logging_hook)
        else:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op,
                eval_metric_ops=eval_metric_ops)

    def _mixup_batch(self, x, y, ratio):
        batch_size = tf.shape(x)[0]
        indices = tf.random.shuffle(tf.range(batch_size, dtype=tf.int32))
        mixed_x = ratio * x + (1 - ratio) * tf.gather(x, indices)
        y_a, y_b = y, tf.gather(y, indices)
        return mixed_x, y_a, y_b

    def _mixup_loss(self, loss, pred, y_a, y_b, ratio):
        return ratio * loss(pred, y_a) + (1 - ratio) * loss(pred, y_b)

    def _init_tf_estimator(self):
        """Init tensorflow estimator."""
        sess_config = self._init_session_config()
        if vega.is_gpu_device():
            self._init_gpu_estimator(sess_config)
        elif vega.is_npu_device():
            self._init_npu_estimator(sess_config)

    def _init_tf_session(self):
        sess_config = self._init_session_config()
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.sess = tf.compat.v1.Session(config=sess_config)

    def _init_session_config(self):
        sess_config = self._init_gpu_session_config() if vega.is_gpu_device() else \
            self._init_npu_session_config()
        return sess_config

    def _init_logging_hook(self):
        logging_hook = []
        if self.horovod:
            import horovod.tensorflow as hvd
            logging_hook += [hvd.BroadcastGlobalVariablesHook(0)]
        return logging_hook

    def _init_gpu_estimator(self, sess_config):
        """Init tensorflow estimator."""
        distribution = None
        if self.horovod:
            distribution = tf.contrib.distribute.MirroredStrategy()
        config = tf.estimator.RunConfig(model_dir=self.get_local_worker_path(),
                                        save_checkpoints_steps=self.config.save_steps,
                                        log_step_count_steps=self.config.train_report_steps,
                                        session_config=None if distribution else sess_config,
                                        train_distribute=distribution)
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn, config=config)

    def _init_npu_estimator(self, sess_config):
        from npu_bridge.estimator.npu.npu_config import NPURunConfig
        from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
        model_dir = self.get_local_worker_path()
        config = NPURunConfig(model_dir=model_dir,
                              save_checkpoints_steps=self.config.save_steps,
                              log_step_count_steps=self.config.train_report_steps,
                              session_config=sess_config,
                              enable_data_pre_proc=True,
                              iterations_per_loop=1)
        self.estimator = NPUEstimator(model_fn=self.model_fn,
                                      config=config)

    def _init_gpu_session_config(self):
        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        return sess_config

    def _init_npu_session_config(self):
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
        from npu_bridge import npu_init
        sess_config = tf.ConfigProto()
        sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        if self.use_amp:
            custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["use_off_line"].b = True
        return sess_config
