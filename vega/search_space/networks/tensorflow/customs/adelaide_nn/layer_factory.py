# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Different custom layers."""
import tensorflow as tf

OPS = {
    'none': lambda C, stride, affine, repeats=1, data_format='channels_first': Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine, repeats=1, data_format='channels_first': AveragePooling2D(
        3, strides=stride, padding='same', data_format=data_format),
    'max_pool_3x3': lambda C, stride, affine, repeats=1, data_format='channels_first': MaxPooling2D(
        3, strides=stride, padding='same', data_format=data_format),
    'global_average_pool': lambda C, stride, affine, repeats=1, data_format='channels_first': GAPConv1x1(C),
    'skip_connect': lambda C, stride, affine, repeats=1, data_format='channels_first': Identity() if stride == 1
    else FactorizedReduce(C, affine),
    'sep_conv_3x3': lambda C, stride, affine, repeats=1, data_format='channels_first': SepConv(
        C, 3, stride, 1, affine=affine, repeats=repeats),
    'sep_conv_5x5': lambda C, stride, affine, repeats=1, data_format='channels_first': SepConv(
        C, 5, stride, 2, affine=affine, repeats=repeats),
    'sep_conv_7x7': lambda C, stride, affine, repeats=1, data_format='channels_first': SepConv(
        C, 7, stride, 3, affine=affine, repeats=repeats),
    'dil_conv_3x3': lambda C, stride, affine, repeats=1, data_format='channels_first': DilConv(
        C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine, repeats=1, data_format='channels_first': DilConv(
        C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine, repeats=1, data_format='channels_first': conv_7x1_1x7(C, stride, affine),
    'conv1x1': lambda C, stride, affine, repeats=1, data_format='channels_first': Conv(C, 1, stride, affine=affine),
    'conv3x3': lambda C, stride, affine, repeats=1, data_format='channels_first': Conv(C, 3, stride, affine=affine),
    'conv5x5': lambda C, stride, affine, repeats=1, data_format='channels_first': Conv(C, 5, stride, affine=affine),
    'conv7x7': lambda C, stride, affine, repeats=1, data_format='channels_first': Conv(C, 7, stride, affine=affine),
    'conv3x3_dil2': lambda C, stride, affine, repeats=1, data_format='channels_first': Conv(
        C, 3, stride, affine=affine, dilation_rate=2),
    'conv3x3_dil3': lambda C, stride, affine, repeats=1, data_format='channels_first': Conv(
        C, 3, stride, affine=affine, dilation_rate=3),
    'conv3x3_dil12': lambda C, stride, affine, repeats=1, data_format='channels_first': Conv(
        C, 3, stride, affine=affine, dilation_rate=12),
    'sep_conv_3x3_dil3': lambda C, stride, affine, repeats=1, data_format='channels_first': SepConv(
        C, 3, stride, affine=affine, dilation_rate=3, repeats=repeats),
    'sep_conv_5x5_dil6': lambda C, stride, affine, repeats=1, data_format='channels_first': SepConv(
        C, 5, stride, affine=affine, dilation_rate=6, repeats=repeats)
}


def resize_bilinear(x, size):
    """Bilinear interplotation."""
    x = tf.image.resize_bilinear(tf.transpose(x, [0, 2, 3, 1]), size=size, align_corners=True)
    x = tf.transpose(x, [0, 3, 1, 2])
    return x


class AveragePooling2D(object):
    """Class of averagepooling2d."""

    def __init__(self, kernel_size, strides, padding, data_format):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

    def __call__(self, x, training):
        """Forward function of avgpooling2d."""
        return tf.layers.average_pooling2d(x, 3, strides=self.strides, padding=self.padding,
                                           data_format=self.data_format)


class MaxPooling2D(object):
    """Class of maxpooling2d."""

    def __init__(self, kernel_size, strides, padding, data_format):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

    def __call__(self, x, training):
        """Forward function of maxpooling2d."""
        return tf.layers.average_pooling2d(x, 3, strides=self.strides, padding=self.padding,
                                           data_format=self.data_format)


class Conv(object):
    """Class of convolution."""

    def __init__(self, out_planes, kernel_size, strides=1, use_bias=False,
                 dilation_rate=1, affine=True, data_format='channels_first'):
        """Init conv."""
        super(Conv, self).__init__()
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.dilation_rate = dilation_rate
        self.affine = affine
        self.data_format = data_format

    def __call__(self, x, training):
        """Forward function of conv."""
        x = tf.layers.conv2d(x, self.out_planes, kernel_size=self.kernel_size,
                             strides=self.strides, padding='same',
                             dilation_rate=self.dilation_rate,
                             use_bias=self.use_bias, data_format=self.data_format)
        x = tf.layers.batch_normalization(x, axis=1 if self.data_format == 'channels_first' else 3,
                                          trainable=self.affine, training=training)
        x = tf.nn.relu(x)
        return x


class conv_7x1_1x7(object):
    """Class of 7x1 and 1x7 convolution.

    :param desc: description of conv_7x1_1x7
    :type desc: Config
    """

    def __init__(self, C_out, strides, affine=True, data_format='channels_first'):
        """Init conv_7x1_1x7."""
        super(conv_7x1_1x7, self).__init__()
        self.strides = strides
        self.C_out = C_out
        self.affine = affine
        self.data_format = data_format

    def __call__(self, x, training):
        """Forward function of conv_7x1_1x7."""
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, self.channel_out, (1, 7), strides=(1, self.stride), padding='same',
                             use_bias=False, data_format=self.data_format)
        x = tf.layers.conv2d(x, self.channel_out, (7, 1), strides=(self.stride, 1), padding='same',
                             use_bias=False, data_format=self.data_format)
        x = tf.layers.batch_normalization(x, axis=1 if self.data_format == 'channels_first' else 3,
                                          trainable=self.affine, training=training)
        return x


class GAPConv1x1(object):
    """Global Average Pooling + conv1x1."""

    def __init__(self, C_out):
        """Construct GAPConv1x1 class.

        :param C_in: input channel
        :param C_out: output channel
        """
        super(GAPConv1x1, self).__init__()
        self.C_out = C_out

    def forward(self, x, training):
        """Do an inference on GAPConv1x1.

        :param x: input tensor
        :return: output tensor
        """
        size = x.get_shape()[2:]
        out = tf.reduce_mean(x, [-2, -1], keepdims=True)
        out = Conv(x, self.C_out, 1, strides=1, use_bias=False,
                   dilation_rate=1, affine=True, data_format='channels_first')(x, training)
        out = resize_bilinear(out, size)
        return out


class DilConv(object):
    """Class of Dilation Convolution."""

    def __init__(self, C_out, kernel_size, strides,
                 dilation_rate, affine=True, data_format='channels_first'):
        super(DilConv, self).__init__()
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.affine = affine
        self.data_format = data_format

    def __call__(self, x, training):
        """Forward function of DilatedConv."""
        desc = self.desc
        x = tf.nn.relu(x)
        x = tf.layers.separable_conv2d(x, self.C_out, self.kernel_size, strides=self.strides,
                                       dilation_rate=self.dilation_rate,
                                       padding='same', use_bias=False, data_format=self.data_format)
        x = tf.layers.batch_normalization(x, axis=1 if desc.data_format == 'channels_first' else 3,
                                          trainable=self.affine, fused=True, training=training)
        return x


class SepConv(object):
    """Class of Separated Convolution."""

    def __init__(self, C_out, kernel_size, strides,
                 dilation_rate=1, affine=True, repeats=1,
                 data_format='channels_first'):
        super(SepConv, self).__init__()
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.affine = affine
        self.repeats = repeats
        self.data_format = data_format

    def basic_op(self, x, training):
        """Sepconv basic op."""
        x = tf.layers.separable_conv2d(x, self.C_out, self.kernel_size, strides=self.strides,
                                       dilation_rate=self.dilation_rate,
                                       padding='same', use_bias=False, data_format=self.data_format)
        x = tf.layers.batch_normalization(x, axis=1 if self.data_format == 'channels_first' else 3,
                                          trainable=self.affine, training=training)
        x = tf.nn.relu(x)
        return x

    def __call__(self, x, training):
        """Forward function of SeparatedConv."""
        for idx in range(self.repeats):
            x = self.basic_op(x, training)
        return x


class Identity(object):
    """Class of Identity operation."""

    def __init__(self):
        """Init Identity."""
        super(Identity, self).__init__()

    def __call__(self, x, training):
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

    def __call__(self, x, training):
        """Forward Function fo Zero."""
        if self.stride == 1:
            return tf.zeros_like(x)
        if self.data_format == 'channels_first':
            return tf.zeros_like(x)[:, :, ::self.stride, ::self.stride]
        else:
            return tf.zeros_like(x)[:, ::self.stride, ::self.stride, :]


class FactorizedReduce(object):
    """Class of Factorized Reduce operation."""

    def __init__(self, channel_out, affine=True, data_format='channels_first'):
        """Init FactorizedReduce."""
        super(FactorizedReduce, self).__init__()
        if channel_out % 2 != 0:
            raise Exception('channel_out must be divided by 2.')
        self.affine = affine
        self.channel_out = channel_out

    def __call__(self, x, training):
        """Forward function of FactorizedReduce."""
        x = tf.nn.relu(x)
        x2 = tf.identity(x[:, :, 1:, 1:] if self.data_format == 'channels_first' else x[:, 1:, 1:, :])
        axis = 1 if self.data_format == 'channels_first' else 3
        out1 = tf.layers.conv2d(x, self.channel_out // 2, 1, strides=2, padding='same',
                                use_bias=False, data_format=self.data_format)
        out2 = tf.layers.conv2d(x2, self.channel_out // 2, 1, strides=2, padding='same',
                                use_bias=False, data_format=self.data_format)
        out = tf.concat([out1, out2], axis=axis)
        out = tf.layers.batch_normalization(out, axis=axis, trainable=self.affine,
                                            fused=True, training=training)
        return out
