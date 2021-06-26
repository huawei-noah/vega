# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Inference of spatiotemporal model."""
import itertools

import tensorflow as tf

from vega.datasets import Adapter
from vega.datasets.common.spatiotemporal import SpatiotemporalDataset
from vega.metrics.tensorflow.forecast import RMSE
from vega.model_zoo import ModelZoo


def _init_tf_estimator(desc_file, model_dir):
    """Init estimator of gpu evaluator used in tf backend."""
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(
        model_dir=model_dir, session_config=sess_config)
    model = ModelZoo().get_model(desc_file)

    def _model_fn(features, labels, mode):
        """Model function of gpu evaluator."""
        model.training = False
        logits = model(features)
        logits = tf.cast(logits, tf.float32)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)
        else:
            eval_metric_ops = RMSE()(logits, labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=tf.log(1.0), train_op=None,
                                              eval_metric_ops=eval_metric_ops)

    return tf.estimator.Estimator(model_fn=_model_fn, config=config)


def predict(data_path, desc_file, pretained_model_dir=None):
    """Predict Spatiotemporal."""
    dataset = SpatiotemporalDataset(
        mode='test', **dict(data_path=data_path, n_his=12, n_pred=4))
    valid_loader = Adapter(dataset).loader
    estimator = _init_tf_estimator(desc_file, pretained_model_dir)
    eval_metrics = estimator.evaluate(
        input_fn=valid_loader.input_fn, steps=len(valid_loader))
    predictions = list(itertools.islice(
        estimator.predict(input_fn=valid_loader.input_fn), 10))
    print(eval_metrics)
    y_pred = predictions[0].reshape(-1) * dataset.std + dataset.mean
    print(y_pred)
    print("mean: {:.2f}, std: {:.2f}".format(dataset.mean, dataset.std))
