# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined operations."""
import tensorflow as tf


class ReluConvBn(object):
    """Class of ReLU + Conv + BN.

    :param desc: description of ReluConvBn
    :type desc: Config
    """

    def __init__(self, desc):
        super(ReluConvBn, self).__init__()
        self.affine = desc.get('affine', True)
        self.desc = desc

    def __call__(self, x, training):
        """Forward function fo ReluConvBn."""
        desc = self.desc
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, desc.channel_out, desc.kernel_size, strides=desc.stride,
                             padding='same', use_bias=False, data_format=desc.data_format)
        x = tf.layers.batch_normalization(x, axis=1 if desc.data_format == 'channels_first' else 3,
                                          trainable=self.affine, fused=True, training=training)
        return x


class DilatedConv(object):
    """Class of Dilation Convolution.

    :param desc: description of DilatedConv
    :type desc: Config
    """

    def __init__(self, desc):
        super(DilatedConv, self).__init__()
        self.affine = desc.get('affine', True)
        self.desc = desc

    def __call__(self, x, training):
        """Forward function of DilatedConv."""
        desc = self.desc
        x = tf.nn.relu(x)
        x = tf.layers.separable_conv2d(x, desc.channel_in, desc.kernel_size, strides=desc.stride,
                                       padding=desc.padding, dilation_rate=desc.dilation,
                                       use_bias=False, data_format=desc.data_format)
        x = tf.layers.conv2d(x, desc.channel_out, kernel_size=1, strides=1, padding=desc.padding,
                             use_bias=False, data_format=desc.data_format)
        x = tf.layers.batch_normalization(x, axis=1 if desc.data_format == 'channels_first' else 3,
                                          trainable=self.affine, fused=True, training=training)
        return x


class SeparatedConv(object):
    """Class of Separated Convolution.

    :param desc: description of SeparatedConv
    :type desc: Config
    """

    def __init__(self, desc):
        super(SeparatedConv, self).__init__()
        self.desc = desc
        self.affine = desc.get('affine', True)

    def __call__(self, x, training):
        """Forward function of SeparatedConv."""
        desc = self.desc
        x = tf.nn.relu(x)
        x = tf.layers.separable_conv2d(x, desc.channel_in, desc.kernel_size, strides=desc.stride,
                                       padding=desc.padding, use_bias=False, data_format=desc.data_format)
        x = tf.layers.conv2d(x, desc.channel_in, kernel_size=1, strides=1, padding=desc.padding,
                             use_bias=False, data_format=desc.data_format)
        x = tf.layers.batch_normalization(x, axis=1 if desc.data_format == 'channels_first' else 3,
                                          trainable=self.affine, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.separable_conv2d(x, desc.channel_in, desc.kernel_size,
                                       padding=desc.padding, use_bias=False, data_format=desc.data_format)
        x = tf.layers.conv2d(x, desc.channel_out, kernel_size=1, strides=1, padding=desc.padding,
                             use_bias=False, data_format=desc.data_format)
        x = tf.layers.batch_normalization(x, axis=1 if desc.data_format == 'channels_first' else 3,
                                          trainable=self.affine, training=training)
        return x


class Identity(object):
    """Class of Identity operation."""

    def __init__(self):
        """Init Identity."""
        super(Identity, self).__init__()

    def __call__(self, x):
        """Forward function of Identity."""
        return tf.identity(x)


class Zero(object):
    """Class of Zero operation.

    :param desc: description of Zero
    :type desc: Config
    """

    def __init__(self, desc):
        """Init Zero."""
        super(Zero, self).__init__()
        self.stride = desc.stride
        self.data_format = desc.data_format

    def __call__(self, x):
        """Forward Function fo Zero."""
        if self.stride == 1:
            return tf.zeros_like(x)
        if self.data_format == 'channels_first':
            return tf.zeros_like(x)[:, :, ::self.stride, ::self.stride]
        else:
            return tf.zeros_like(x)[:, ::self.stride, ::self.stride, :]


class FactorizedReduce(object):
    """Class of Factorized Reduce operation.

    :param desc: description of FactorizedReduce
    :type desc: Config
    """

    def __init__(self, desc):
        """Init FactorizedReduce."""
        super(FactorizedReduce, self).__init__()
        if desc.channel_out % 2 != 0:
            raise Exception('channel_out must be divided by 2.')
        self.affine = desc.get('affine', True)
        self.desc = desc

    def __call__(self, x, training):
        """Forward function of FactorizedReduce."""
        desc = self.desc
        x = tf.nn.relu(x)
        x2 = tf.identity(x[:, :, 1:, 1:] if desc.data_format == 'channels_first' else x[:, 1:, 1:, :])
        axis = 1 if desc.data_format == 'channels_first' else 3
        out1 = tf.layers.conv2d(x, self.desc.channel_out // 2, 1, strides=2, padding='same',
                                use_bias=False, data_format=desc.data_format)
        out2 = tf.layers.conv2d(x2, self.desc.channel_out // 2, 1, strides=2, padding='same',
                                use_bias=False, data_format=desc.data_format)
        out = tf.concat([out1, out2], axis=axis)
        out = tf.layers.batch_normalization(out, axis=axis, trainable=self.affine,
                                            fused=True, training=training)
        return out


def drop_path(x, prob):
    """Drop path operation.

    :param x: input feature map
    :type x: torch tensor
    :param prob: dropout probability
    :type prob: float
    :return: output feature map after dropout
    :rtype: torch tensor
    """
    if prob <= 0.:
        return x
    keep = 1. - prob

    bernoulli_random = tf.random.uniform([x.get_shape(0), 1, 1, 1])
    mask = bernoulli_random < keep
    x = tf.div(x, keep)
    x = tf.mul(x, mask)
    return x
