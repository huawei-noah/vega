# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""CARS trainer."""
import logging

import tensorflow as tf
import tf_slim as slim
from object_detection.core import standard_fields as fields
from object_detection.utils import variables_helper

from vega.common import ClassFactory, ClassType
from vega.trainer.callbacks import Callback
from .tf_optimizer import TFOptimizer


@ClassFactory.register(ClassType.CALLBACK)
class FasterRCNNTrainerCallback(Callback):
    """A special callback for FasterRCNNTrainer."""

    disable_callbacks = ["ModelStatistics"]

    def model_fn(self, features, labels, mode):
        """Define Faster R-CNN model_fn used by TensorFlow Estimator."""
        logging.info('Faster R-CNN model function action')
        self.model = self.trainer.model
        self.config = self.trainer.config
        predict_result_dict = self.model(
            features, labels, mode == tf.estimator.ModeKeys.TRAIN)

        self.fine_tune_checkpoint_type = self.config.fine_tune_checkpoint_type
        self.load_all_detection_checkpoint_vars = True
        asg_map = self.model.restore_map(
            fine_tune_checkpoint_type=self.fine_tune_checkpoint_type,
            load_all_detection_checkpoint_vars=(
                self.load_all_detection_checkpoint_vars))

        self.fine_tune_checkpoint = self.config.fine_tune_checkpoint
        available_var_map = (
            variables_helper.get_variables_available_in_checkpoint(
                asg_map,
                self.fine_tune_checkpoint,
                include_global_step=False))
        tf.train.init_from_checkpoint(self.fine_tune_checkpoint,
                                      available_var_map)

        losses_dict = self.model.loss(
            predict_result_dict, features[fields.InputDataFields.true_image_shape])
        losses = [loss_tensor for loss_tensor in losses_dict.values()]
        total_loss = tf.add_n(losses, name='total_loss')
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            self.optimizer, self.optimizer_summary_vars = TFOptimizer(
                self.config.optimizer).get_real_optimizer(global_step)
            trainable_variables = None
            trainable_variables = slim.filter_variables(
                tf.trainable_variables())
            clip_gradients_value = None
            summaries = None
            train_op = slim.optimizers.optimize_loss(
                loss=total_loss,
                global_step=global_step,
                learning_rate=None,
                clip_gradients=clip_gradients_value,
                optimizer=self.optimizer,
                update_ops=self.model.updates(),
                variables=trainable_variables,
                summaries=summaries,
                name='')  # Preventing scope prefix on all variables.

        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.valid_metrics(predict_result_dict, labels)
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op,
                                          eval_metric_ops=eval_metric_ops)
