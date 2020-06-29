# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Simple CNN Network."""
import tensorflow as tf
from vega.search_space.networks import NetTypes, NetworkFactory


@NetworkFactory.register(NetTypes.BACKBONE)
class SimpleCnn(object):
    """Simple Cnn Network of classification.

    :param desc: SimpleCnn description
    :type desc: Config
    """

    def __init__(self, desc):
        self.num_class = desc.num_class
        self.fp16 = desc.get('fp16', False)
        self.blocks = desc.blocks
        self.channels = desc.channels
        self.conv = tf.layers.conv2d
        self.pool = tf.layers.max_pooling2d
        self.flat = tf.reshape
        self.bn = tf.layers.batch_normalization
        self.fc = tf.layers.dense
        self.dropout = tf.layers.dropout
        self.relu = tf.nn.relu

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=tf.float32, *args, **kwargs):
        """Convert variable of operation into tf.float16."""
        if self.fp16 and dtype == tf.float16:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self):
        """Define Scope of model variable."""
        return tf.variable_scope('simple_cnn', custom_getter=self._custom_dtype_getter)

    def __call__(self, inputs, training):
        """Call Simple Cnn forward function."""
        with self._model_variable_scope():
            inputs = self.conv(inputs, filters=32, kernel_size=[3, 3], padding='same', activation=self.relu)
            inputs = self.pool(inputs, pool_size=[2, 2], strides=2)
            for _ in range(self.blocks):
                inputs = self._block(self.channels, inputs, training)
            inputs = self.pool(inputs, pool_size=[2, 2], strides=2)
            inputs = self.conv(inputs, filters=64, kernel_size=[3, 3], padding='same', activation=self.relu)
            inputs = self.conv(inputs, filters=64, kernel_size=[8, 8], padding='valid', activation=self.relu)
            inputs = self.flat(inputs, [-1, 64])
            logits = self.fc(inputs, units=self.num_class)
            logits = tf.identity(logits, 'logits')
            return logits

    def _block(self, channels, inputs, training):
        """Define basic block in simple cnn."""
        inputs = self.conv(inputs, filters=channels, kernel_size=[3, 3], padding='same')
        inputs = self.bn(inputs, training=training, scale=True, fused=True)
        inputs = self.relu(inputs)
        return inputs
