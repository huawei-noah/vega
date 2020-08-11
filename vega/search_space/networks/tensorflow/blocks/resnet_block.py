# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ResNet Block."""
import tensorflow as tf
from vega.algorithms.compression.quant_ea.utils.tensorflow.quant_conv import QuantConv


def batch_norm(inputs, training, data_format):
    """Initialize a standard batch normalization."""
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=0.997, epsilon=1e-5, center=True,
        scale=True, training=training, fused=True)


def conv_same_padding(inputs, filters, kernel_size, strides, data_format):
    """Convolution operation with type same padding."""
    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding='SAME', use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


def small_initial_block(inputs, filters, data_format):
    """Build small initial block for ResNet."""
    inputs = conv_same_padding(inputs=inputs, filters=filters, kernel_size=3,
                               strides=1, data_format=data_format)
    return inputs


def initial_block(inputs, filters, data_format):
    """Build normal initial block for ResNet."""
    inputs = conv_same_padding(inputs=inputs, filters=filters, kernel_size=7,
                               strides=2, data_format=data_format)
    inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=2,
                                     padding='SAME', data_format=data_format)
    return inputs


def basic_block(inputs, filters, training, strides, data_format):
    """Build basic block for ResNet."""
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    chn_axis = 1 if data_format == 'channels_first' else 3
    in_filters = inputs.get_shape()[chn_axis]
    if strides != 1 or in_filters != filters:
        shortcut = conv_same_padding(inputs=inputs, filters=filters, kernel_size=1,
                                     strides=strides, data_format=data_format)

    inputs = conv_same_padding(inputs=inputs, filters=filters, kernel_size=3,
                               strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv_same_padding(inputs=inputs, filters=filters, kernel_size=3,
                               strides=1, data_format=data_format)

    return inputs + shortcut


def bottleneck_block(inputs, filters, training, strides, data_format):
    """Build bottleneck block for ResNet."""
    p = 4
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    chn_axis = 1 if data_format == 'channels_first' else 3
    in_filters = inputs.get_shape()[chn_axis]
    if strides != 1 or in_filters != p * filters:
        shortcut = conv_same_padding(inputs=inputs, filters=p * filters, kernel_size=1,
                                     strides=strides, data_format=data_format)

    inputs = conv_same_padding(inputs=inputs, filters=filters, kernel_size=1,
                               strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv_same_padding(inputs=inputs, filters=filters, kernel_size=3,
                               strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv_same_padding(inputs=inputs, filters=p * filters, kernel_size=1,
                               strides=1, data_format=data_format)

    return inputs + shortcut


def _prune_basic_block(x, planes, inner_plane, training, data_format, name, strides=1):
    """Forward function of BasicBlock."""
    shortcut = x
    axis = 1 if data_format == 'channels_first' else 3
    in_plane = x.get_shape()[axis]
    x = tf.layers.conv2d(x, inner_plane, 3, padding='same', strides=strides,
                         use_bias=False, data_format=data_format, name=name + '/conv_1')
    x = tf.layers.batch_normalization(x, axis=axis, name=name + '/bn_1', training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, planes, 3, padding='same', strides=1, use_bias=False,
                         data_format=data_format, name=name + '/conv_2')
    x = tf.layers.batch_normalization(x, axis=axis, name=name + '/bn_2', training=training)
    x = tf.nn.relu(x)
    if strides != 1 or in_plane != planes:
        shortcut = tf.layers.conv2d(shortcut, planes, 1, padding='same', strides=strides, use_bias=False,
                                    data_format=data_format, name=name + '/conv_shortcut')
        shortcut = tf.layers.batch_normalization(shortcut, axis=axis, name=name + '/bn_shortcut', training=training)
        shortcut = tf.nn.relu(shortcut)

    return x + shortcut


def _quant_basic_block(x, planes, training, data_format, name, strides=1, quant_info=None):
    """Forward function of BasicBlock."""
    shortcut = x
    axis = 1 if data_format == 'channels_first' else 3
    in_plane = x.get_shape()[axis]

    if quant_info is not None:
        quant_conv = QuantConv(planes, 3, name=name + '/conv_1', strides=strides,
                               padding='same', groups=1, use_bias=False, data_format=data_format)
        quant_conv.quant_config(quant_info=quant_info, name=name + '/conv_1')
        x = quant_conv(x)
    else:
        x = tf.layers.conv2d(x, planes, 3, padding='same', strides=strides, use_bias=False,
                             data_format=data_format, name=name + '/conv_1')
    x = tf.layers.batch_normalization(x, axis=axis, name=name + '/bn_1', training=training)
    x = tf.nn.relu(x)

    if quant_info is not None:
        quant_conv = QuantConv(planes, 3, name=name + '/conv_2', strides=1,
                               padding='same', groups=1, use_bias=False, data_format=data_format)
        quant_conv.quant_config(quant_info=quant_info, name=name + '/conv_2')
        x = quant_conv(x)
    else:
        x = tf.layers.conv2d(x, planes, 3, padding='same', strides=1, use_bias=False,
                             data_format=data_format, name=name + '/conv_2')
    x = tf.layers.batch_normalization(x, axis=axis, name=name + '/bn_2', training=training)

    if strides != 1 or in_plane != planes:
        if quant_info is not None:
            quant_conv = QuantConv(planes, 1, name=name + '/conv_shortcut', strides=strides,
                                   padding='same', groups=1, use_bias=False, data_format=data_format)
            quant_conv.quant_config(quant_info=quant_info, name=name + '/conv_shortcut')
            shortcut = quant_conv(shortcut)
        else:
            shortcut = tf.layers.conv2d(shortcut, planes, 1, padding='same', strides=strides, use_bias=False,
                                        data_format=data_format, name=name + '/conv_shortcut')
        shortcut = tf.layers.batch_normalization(shortcut, axis=axis, name=name + '/bn_shortcut', training=training)
    return tf.nn.relu(x + shortcut)
