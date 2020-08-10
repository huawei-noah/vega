# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Residual Network."""
import tensorflow as tf
from vega.search_space.networks import NetTypes, NetworkFactory
from ..blocks.resnet_block import batch_norm, conv_same_padding
from ..blocks.resnet_block import basic_block, bottleneck_block
from ..blocks.resnet_block import small_initial_block, initial_block


@NetworkFactory.register(NetTypes.BACKBONE)
class ResNetVariant(object):
    """ResNet Variant Class created by code of doublechannel and downsample."""

    def __init__(self, desc):
        self.net_desc = desc
        self.base_depth = desc.base_depth
        self.base_channel = desc.base_channel
        self.bottleneck = False if self.base_depth < 50 else True
        if self.bottleneck:
            self.block_fn = bottleneck_block
        else:
            self.block_fn = basic_block
        self.data_format = desc.get('data_format', 'channels_first')
        self.num_classes = desc.num_classes
        self.fp16 = tf.float16 if desc.get('fp16', False) else tf.float32
        self.doublechannel = desc.doublechannel
        self.downsample = desc.downsample
        if len(self.doublechannel) != len(self.downsample):
            raise ValueError('length of doublechannel must be equal to downsample')

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=tf.float16,
                             *args, **kwargs):
        """Create variables in fp32, then casts to fp16 if necessary."""
        if self.fp16 and dtype == tf.float16:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self):
        """Return a variable scope of created model."""
        return tf.variable_scope('resnet_model_variant',
                                 custom_getter=self._custom_dtype_getter)

    def __call__(self, inputs, training):
        """Call ResNetVariant forward function."""
        with self._model_variable_scope():
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])
            small_input = True if inputs.get_shape()[2] < 64 else False
            if small_input:
                inputs = small_initial_block(inputs, self.base_channel, self.data_format)
            else:
                inputs = initial_block(inputs, self.base_channel, self.data_format)
            inputs = tf.identity(inputs, 'initial_feats')

            inputs = self._forward_resolution_block(inputs, self.doublechannel, self.downsample,
                                                    self.base_channel, training)

            inputs = batch_norm(inputs, training, self.data_format)
            inputs = tf.nn.relu(inputs)

            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            inputs = tf.reduce_mean(inputs, axes, keepdims=True)
            inputs = tf.identity(inputs, 'reduce_mean')

            inputs = tf.squeeze(inputs, axes)
            inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
            inputs = tf.identity(inputs, 'dense')
            return inputs

    def _forward_resolution_block(self, inputs, doublechannel, downsample, num_filters, training):
        """Create resolution block of ResNet."""
        for idx in range(len(doublechannel)):
            num_filters = num_filters if doublechannel[idx] == 0 else num_filters * 2
            strides = 1 if downsample[idx] == 0 else 2
            inputs = self.block_fn(inputs, num_filters, training, strides, self.data_format)
        return inputs
