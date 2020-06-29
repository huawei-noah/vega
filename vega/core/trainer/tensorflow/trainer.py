# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""TensorFlow Trainer."""
import importlib
import json
import logging
import os
import pickle
import time
import copy
from functools import partial
import tensorflow as tf
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.config import Config
from vega.core.common.utils import update_dict
from vega.core.trainer.distributed_worker import DistributedWorker
from vega.core.trainer.utils import WorkerTypes
from vega.search_space.networks import NetworkDesc
from vega.core.common.file_ops import FileOps
from tensorflow.python.estimator import estimator as est
import vega.datasets.tensorflow

try:
    import horovod.tensorflow as hvd
except Exception as e:
    # logging.warning("horovod not been installed, {}".format(str(e)))
    pass


@ClassFactory.register(ClassType.TRAINER)
class Trainer(DistributedWorker):
    """TensorFlow Trainer class.

    :param model: input model, defaults to None
    :type model: tf model, optional
    :param id: id of the model, defaults to None
    :type id: int, optional
    :param hps: hyperparameters, defaults to None
    :type hps: dict, optional
    """

    __worker_id__ = 0

    def __init__(self, model=None, id=None, hps=None, **kwargs):
        super(Trainer, self).__init__(self.cfg)
        self.worker_type = WorkerTypes.TRAINER
        Trainer.__worker_id__ += 1
        if id is not None:
            self._worker_id = id
        else:
            self._worker_id = Trainer.__worker_id__
        self.model = model
        self.horovod = self.cfg.get('horovod', False)

    def _init_model(self):
        """Load model desc from save path and parse to model."""
        model_desc = self.cfg.model_desc
        model = NetworkDesc(model_desc).to_model()
        return model

    def _init_loss(self):
        """Init loss function from tensorflow according to type in config."""
        loss_config = self.cfg.loss.copy()
        loss_name = loss_config.pop('type')
        loss_class = getattr(
            importlib.import_module('tensorflow.losses'),
            loss_name)
        return partial(loss_class, **loss_config)

    def _init_optimizer(self):
        """Init optimizer from tensorflow according to optim type in config."""
        optim_config = self.cfg.optim.copy()
        optim_name = optim_config.pop('type')
        optim_class = getattr(importlib.import_module('tensorflow.train'),
                              optim_name)
        optimizer = optim_class(**optim_config)
        if self.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer)
        return optimizer

    def _init_metric(self):
        """Init metric from tensorflow according to type in config."""
        metric_config = self.cfg.metric.copy()
        metric_name = metric_config.pop('type')
        metric_class = getattr(importlib.import_module('tensorflow.metrics'),
                               metric_name)
        return partial(metric_class, **metric_config)

    def model_fn(self, features, labels, mode):
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
        if self.model is None:
            self.model = self._init_model()
        logits = self.model(features, mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.cast(logits, tf.float32)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'predict': tf.estimator.export.PredictOutput(predictions)
                })
        self.loss = self._init_loss()
        loss = self.loss(logits=logits, labels=labels)
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.optimizer = self._init_optimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = self.optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())

        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            self.metric = self._init_metric()
            eval_metric_ops = {"accuracy": self.metric(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          eval_metric_ops=eval_metric_ops)

    def _init_estimator(self):
        """Init tensorflow estimator."""
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        if self.horovod:
            sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
        config = tf.estimator.RunConfig(model_dir=self.get_local_worker_path(),
                                        save_checkpoints_steps=self.cfg.save_steps,
                                        session_config=sess_config)
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                config=config)

    def _init_dataloader(self):
        """Init dataloader."""
        data_cls = ClassFactory.get_cls(ClassType.DATASET)
        data_cfg = copy.deepcopy(ClassFactory.__configs__.get(ClassType.DATASET))
        data_cfg.pop('type')
        self.train_data, self.valid_data = [
            data_cls(**data_cfg, mode=mode) for mode in ['train', 'val']
        ]

    @property
    def _first_rank(self):
        """Check if the first rank."""
        if self.horovod and hvd.rank() != 0:
            return False
        else:
            return True

    def train_process(self):
        """Whole train process of the TrainWorker specified in config.

        After training, the model and validation results are saved to local_worker_path and s3_path.
        """
        self._init_estimator()
        self._init_dataloader()
        logging_hook = []
        if self.horovod:
            logging_hook += [hvd.BroadcastGlobalVariablesHook(0)]
        train_steps = self.train_data.data_len
        valid_steps = self.valid_data.data_len
        if self.horovod:
            train_steps = train_steps // hvd.size()
            valid_steps = valid_steps // hvd.size()
        start_step = est._load_global_step_from_checkpoint_dir(self.get_local_worker_path())
        for i in range(self.cfg.epochs):
            logging.info('train epoch [{0}/{1}]'.format(i, self.cfg.epochs))
            current_max_step = start_step + train_steps
            start_step = current_max_step
            self.estimator.train(input_fn=self.train_data.input_fn,
                                 max_steps=current_max_step,
                                 hooks=logging_hook)
            eval_results = self.estimator.evaluate(input_fn=self.valid_data.input_fn, steps=valid_steps)
            logging.info(eval_results)
        self.save_backup(eval_results)

    def save_backup(self, performance):
        """Save checkpoints and performance file to backup path.

        :param performance: validated performance
        :type param: float, list or dict
        """
        if self.backup_base_path is None:
            return
        pfm_file = os.path.join(self.get_local_worker_path(), 'performance.txt')
        with open(pfm_file, 'w') as f:
            f.write("{}".format(performance))
        backup_worker_path = FileOps.join_path(self.backup_base_path, self.get_worker_subpath())
        FileOps.copy_folder(self.get_local_worker_path(), backup_worker_path)
