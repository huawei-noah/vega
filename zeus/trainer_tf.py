# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""TensorFlow Trainer."""
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.estimator import estimator as est

from zeus.common.general import General
import zeus
from zeus.metrics.tensorflow.metrics import Metrics
from zeus.trainer_base import TrainerBase
try:
    import horovod.tensorflow as hvd
except Exception:
    pass

if zeus.is_npu_device():
    from npu_bridge.estimator.npu.npu_config import NPURunConfig
    from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
    from npu_bridge.estimator import npu_ops
    from hccl.manage.api import get_local_rank_id
    from hccl.manage.api import get_rank_size
    from hccl.manage.api import get_rank_id
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from zeus.common import FileOps, init_log
from zeus.trainer.modules.losses import Loss
from zeus.trainer.modules.lr_schedulers import LrScheduler
from zeus.trainer.modules.optimizer import Optimizer
from zeus.tf_utils import TFVariables


class TrainerTf(TrainerBase):
    """Trainer tensorflow class."""

    def build(self):
        """Build the trainer by assembling the necessary components."""
        super().build()

        # Some trainer has different train batch size from valid batch
        self.train_metrics = None
        self.valid_metrics = self._init_metrics()
        self._init_horovod_setting()

    def train(self, inputs, labels):
        """Train model."""
        feed_dict = {}
        with self.graph.as_default():
            if self.gpu_nums >= 1:
                input_split = [[]] * self.gpu_nums
                shape_split = inputs[0].shape[0] // self.gpu_nums

                for j in range(self.gpu_nums):
                    for i in range(len(inputs)):
                        input_split = inputs[i][j * shape_split: (j + 1) * shape_split]
                        feed_dict.update({self.inputs[j][i]: input_split})

                for j in range(self.gpu_nums):
                    for i in range(len(labels)):
                        input_split = labels[i][j * shape_split: (j + 1) * shape_split]
                        feed_dict.update({self.labels[j][i]: input_split})

            else:
                for i in range(len(inputs)):
                    feed_dict.update({self.inputs[i]: inputs[i]})

                for i in range(len(labels)):
                    feed_dict.update({self.labels[i]: labels[i]})

            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)

            return loss

    def predict(self, input):
        """Inference model."""
        with self.graph.as_default():
            feed_dict = {self.input: input}
            out = self.sess.run(self.logits, feed_dict)
            return out

    def save(self, file_name):
        """Save model."""
        with self.graph.as_default():
            self.actor_var.save_weights(file_name + ".npz")

        return file_name + ".npz"

    def load(self, model_name, by_name):
        """Load model."""
        with self.graph.as_default():
            self.actor_var.set_weights_with_npz(model_name)

    def set_weights(self, weights):
        """Set weight with memory tensor."""
        with self.graph.as_default():
            self.actor_var.set_weights(weights)

    def get_weights(self):
        """Get the weights."""
        with self.graph.as_default():
            return self.actor_var.get_weights()

    def init_train_op(self):
        """Init Train Op."""
        with self.graph.as_default():
            self._init_train_op()

    def _set_default_funcs(self):
        self.model_fn = self._default_model_fn
        self.train_input_fn = self._default_train_input_fn
        self.valid_input_fn = self._default_valid_input_fn

    def _set_condition(self):
        self._init_tf_session()
        self._init_distributed_setting()
        self._init_tf_estimator()

    def _train_epoch(self):
        self.estimator.train(input_fn=self.train_input_fn,
                             steps=len(self.train_loader),
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
        if not self.distributed:
            return
        if zeus.is_npu_device():
            self.npu_init = npu_ops.initialize_system()
            self.npu_shutdown = npu_ops.shutdown_system()
            self.sess.run(self.npu_init)
        self._world_size = hvd.size() if zeus.is_gpu_device() else get_rank_size()
        self._rank_id = hvd.rank() if zeus.is_gpu_device() else get_rank_id()
        self._local_rank_id = hvd.local_rank() if zeus.is_gpu_device() else get_local_rank_id()

    def _build_multigpu_train_op(self, num_gpus):
        with self.graph.as_default(), tf.device('/gpu:0'):
            tower_grads = []
            self.inputs = []
            self.labels = []
            opt = Optimizer()()
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        # tf.get_variable_scope().reuse_variables()
                        inputs = self._create_tensor(self.loss_input['inputs'])
                        labels = self._create_tensor(self.loss_input['labels'])
                        input = inputs[0]
                        model_output = self.model(input)

                        loss = Loss()()
                        loss = loss(model_output, labels)

                        # Calculate the gradients for the batch of data on this tower.
                        varlist = [x for x in tf.trainable_variables() if x.name.startswith('tower_%d' % i)]
                        grads = opt.compute_gradients(loss, varlist)

                        tower_grads.append(grads)
                        if i == 0:
                            self.actor_var = TFVariables(model_output, self.sess)
                            self.input = input
                            self.logits = model_output
                            self.loss = loss

                        self.inputs.append(inputs)
                        self.labels.append(labels)

            grads = self._average_gradients(tower_grads)
            self.train_op = opt.apply_gradients(grads)
            self.sess.run(tf.initialize_all_variables())

    def _average_gradients(self, tower_grads):
        avg_grads = []

        for grad_and_vars in zip(*tower_grads):
            grads = []

            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            all_grad = tf.concat(grads, 0)
            avg_grad = tf.reduce_mean(all_grad, 0, keep_dims=False)

            for _, v in grad_and_vars:
                grad_and_var = (avg_grad, v)
                avg_grads.append(grad_and_var)

        return avg_grads

    def _create_tensor(self, tensor_list):
        ret_list = []

        for tensor in tensor_list:
            tensor_type = tensor['type']
            tensor_shape = tensor['shape']
            tensor_name = tensor['name']

            if type(tensor_shape) is list:
                tf_tensor = tf.placeholder(tensor_type, name=tensor_name,
                                           shape=(None, ) + tuple(tensor_shape))
            else:
                tf_tensor = tf.placeholder(tensor_type, name=tensor_name,
                                           shape=(None, tensor_shape))
            ret_list.append(tf_tensor)

        return ret_list

    def _init_train_op(self):
        self.train_count = 0
        self.train_time = 0.
        import os
        if self.gpu_nums >= 1:
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
                with tf.name_scope('tower_0') as scope:
                    self._build_train_op()
            else:
                self._build_multigpu_train_op(self.gpu_nums)
        else:
            self._build_train_op()

    def _build_train_op(self):
        self.inputs = self._create_tensor(self.loss_input['inputs'])
        self.labels = self._create_tensor(self.loss_input['labels'])

        self.input = self.inputs[0]
        logits = self.model(self.input)
        self.logits = logits
        self.actor_var = TFVariables(logits, self.sess)

        loss = Loss()()
        self.loss = loss(logits, self.labels)

        self.optimizer = Optimizer()(distributed=self.distributed)
        grads_and_var = self.optimizer.compute_gradients(self.loss)
        grads, var = zip(*grads_and_var)
        grads_and_var = list(zip(grads, var))
        self.train_op = self.optimizer.apply_gradients(grads_and_var)
        self.sess.run(tf.initialize_all_variables())

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
        logits = self.model(features)
        logits = tf.cast(logits, tf.float32)
        if hasattr(self.model, 'add_loss'):
            loss_cls = Loss()()
            self.model.add_loss(loss_cls)
            self.loss = self.model.overall_loss()
        else:
            self.loss = Loss()()
        loss = self.loss(logits, labels)
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.compat.v1.train.get_or_create_global_step()
            epoch = tf.cast(global_step, tf.float32) / tf.cast(len(self.train_loader), tf.float32)
            self.optimizer = Optimizer()(distributed=self.distributed)
            self.lr_scheduler = LrScheduler()(optimizer=self.optimizer)
            self.lr_scheduler.step(epoch)
            if self.distributed:
                self.optimizer = Optimizer.set_distributed(self.optimizer)
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            loss_scale = self.config.loss_scale if self.use_amp else 1
            minimize_op = self.optimizer.step(loss, loss_scale, global_step)
            train_op = tf.group(minimize_op, update_ops)

        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.valid_metrics(logits, labels)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          eval_metric_ops=eval_metric_ops)

    def _init_tf_estimator(self):
        """Init tensorflow estimator."""
        sess_config = self._init_session_config()
        if zeus.is_gpu_device():
            self._init_gpu_estimator(sess_config)
        elif zeus.is_npu_device():
            self._init_npu_estimator(sess_config)

    def _init_tf_session(self):
        sess_config = self._init_session_config()
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.sess = tf.compat.v1.Session(config=sess_config)

    def _init_session_config(self):
        sess_config = self._init_gpu_session_config() if zeus.is_gpu_device() else \
            self._init_npu_session_config()
        return sess_config

    def _init_logging_hook(self):
        logging_hook = []
        if zeus.is_gpu_device() and self.distributed:
            logging_hook += [hvd.BroadcastGlobalVariablesHook(0)]
        return logging_hook

    def _init_gpu_estimator(self, sess_config):
        """Init tensorflow estimator."""
        distribution = None
        if not self.distributed and General._parallel and General.devices_per_trainer > 1:
            distribution = tf.contrib.distribute.MirroredStrategy()
        config = tf.estimator.RunConfig(model_dir=self.get_local_worker_path(),
                                        save_checkpoints_steps=self.config.save_steps,
                                        log_step_count_steps=self.config.report_freq,
                                        session_config=None if distribution else sess_config,
                                        train_distribute=distribution)
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                config=config)

    def _init_npu_estimator(self, sess_config):
        model_dir = self.get_local_worker_path()
        config = NPURunConfig(model_dir=model_dir,
                              save_checkpoints_steps=self.config.save_steps,
                              log_step_count_steps=self.config.report_freq,
                              session_config=sess_config,
                              enable_data_pre_proc=True,
                              iterations_per_loop=1)
        self.estimator = NPUEstimator(model_fn=self.model_fn,
                                      config=config)

    def _init_gpu_session_config(self):
        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        if self.distributed:
            sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
        return sess_config

    def _init_npu_session_config(self):
        sess_config = tf.ConfigProto()
        sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        if self.use_amp:
            custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["use_off_line"].b = True

        return sess_config
