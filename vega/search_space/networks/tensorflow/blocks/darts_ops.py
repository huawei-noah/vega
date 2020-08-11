# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined darts operations."""
import tensorflow as tf
from vega.search_space.networks.tensorflow.blocks.operations import *
from vega.search_space.networks import NetTypes, NetworkFactory


@NetworkFactory.register(NetTypes.BLOCK)
class none(object):
    """Class of none.

    :param desc: description of none
    :type desc: Config
    """

    def __init__(self, desc):
        """Init none."""
        super(none, self).__init__()
        self.desc = desc

    def __call__(self, x, training=True):
        """Forward function of none."""
        return Zero(self.desc)(x)


@NetworkFactory.register(NetTypes.BLOCK)
class avg_pool_3x3(object):
    """Class of 3x3 average pooling.

    :param desc: description of avg_pool_3x3
    :type desc: Config
    """

    def __init__(self, desc):
        """Init avg_pool_3x3."""
        super(avg_pool_3x3, self).__init__()
        self.stride = desc.stride
        self.data_format = desc.data_format

    def __call__(self, x, training=True):
        """Forward function of avg_pool_3x3."""
        return tf.layers.average_pooling2d(x, 3, strides=self.stride, padding='same',
                                           data_format=self.data_format)


@NetworkFactory.register(NetTypes.BLOCK)
class max_pool_3x3(object):
    """Class 3x3 max pooling.

    :param desc: description of max_pool_3x3
    :type desc: Config
    """

    def __init__(self, desc):
        """Init max_pool_3x3."""
        super(max_pool_3x3, self).__init__()
        self.stride = desc.stride
        self.data_format = desc.data_format

    def __call__(self, x, training=True):
        """Forward function of max_pool_3x3."""
        return tf.layers.max_pooling2d(x, 3, strides=self.stride, padding='same',
                                       data_format=self.data_format)


@NetworkFactory.register(NetTypes.BLOCK)
class skip_connect(object):
    """Class of skip connect.

    :param desc: description of skip_connect
    :type desc: Config
    """

    def __init__(self, desc):
        """Init skip_connect."""
        super(skip_connect, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        self.desc = desc

    def __call__(self, x, training):
        """Forward function of skip_connect."""
        if self.desc.stride == 1:
            return Identity()(x)
        else:
            return FactorizedReduce(self.desc)(x, training=training)


@NetworkFactory.register(NetTypes.BLOCK)
class sep_conv_3x3(object):
    """Class of 3x3 separated convolution.

    :param desc: description of sep_conv_3x3
    :type desc: Config
    """

    def __init__(self, desc):
        """Init sep_conv_3x3."""
        super(sep_conv_3x3, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 3
        desc.padding = 'same'
        self.desc = desc

    def __call__(self, x, training):
        """Forward function of sep_conv_3x3."""
        return SeparatedConv(self.desc)(x, training=training)


@NetworkFactory.register(NetTypes.BLOCK)
class sep_conv_5x5(object):
    """Class of 5x5 separated convolution.

    :param desc: description of sep_conv_5x5
    :type desc: Config
    """

    def __init__(self, desc):
        """Init sep_conv_5x5."""
        super(sep_conv_5x5, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 5
        desc.padding = 'same'
        self.desc = desc

    def __call__(self, x, training):
        """Forward function of sep_conv_5x5."""
        return SeparatedConv(self.desc)(x, training=training)


@NetworkFactory.register(NetTypes.BLOCK)
class sep_conv_7x7(object):
    """Class of 7x7 separated convolution.

    :param desc: description of sep_conv_7x7
    :type desc: Config
    """

    def __init__(self, desc):
        """Init sep_conv_7x7."""
        super(sep_conv_7x7, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 7
        desc.padding = 'same'
        self.desc = desc

    def __call__(self, x, training):
        """Forward function of sep_conv_7x7."""
        return SeparatedConv(self.desc)(x, training=training)


@NetworkFactory.register(NetTypes.BLOCK)
class dil_conv_3x3(object):
    """Class of 3x3 dilation convolution.

    :param desc: description of dil_conv_3x3
    :type desc: Config
    """

    def __init__(self, desc):
        """Init dil_conv_3x3."""
        super(dil_conv_3x3, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 3
        desc.padding = 'same'
        desc.dilation = 2
        self.desc = desc

    def __call__(self, x, training):
        """Forward function of dil_conv_3x3."""
        return DilatedConv(self.desc)(x, training=training)


@NetworkFactory.register(NetTypes.BLOCK)
class dil_conv_5x5(object):
    """Class of 5x5 dilation convolution.

    :param desc: description of dil_conv_5x5
    :type desc: Config
    """

    def __init__(self, desc):
        """Init dil_conv_5x5."""
        super(dil_conv_5x5, self).__init__()
        desc.channel_in = desc.C
        desc.channel_out = desc.C
        desc.kernel_size = 5
        desc.padding = 'same'
        desc.dilation = 2
        self.desc = desc

    def __call__(self, x, training):
        """Forward function of dil_conv_5x5."""
        return DilatedConv(self.desc)(x, training=training)


@NetworkFactory.register(NetTypes.BLOCK)
class conv_7x1_1x7(object):
    """Class of 7x1 and 1x7 convolution.

    :param desc: description of conv_7x1_1x7
    :type desc: Config
    """

    def __init__(self, desc):
        """Init conv_7x1_1x7."""
        super(conv_7x1_1x7, self).__init__()
        self.stride = desc.stride
        self.channel_out = desc.C
        self.affine = desc.affine
        self.channel_out = channel_out
        self.data_format = desc.data_format

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


@NetworkFactory.register(NetTypes.BLOCK)
class PreOneStem(object):
    """Class of one stem convolution.

    :param desc: description of PreOneStem
    :type desc: Config
    """

    def __init__(self, desc):
        """Init PreOneStem."""
        self._C = desc.C
        self._stem_multi = desc.stem_multi
        self.C_curr = self._stem_multi * self._C
        self.data_format = desc.data_format

    def __call__(self, x, training):
        """Forward function of PreOneStem."""
        x = tf.layers.conv2d(x, self.C_curr, 3, padding='same', use_bias=False, data_format=self.data_format)
        x = tf.layers.batch_normalization(x, axis=1 if self.data_format == 'channels_first' else 3,
                                          fused=True, training=training)
        return x, x


@NetworkFactory.register(NetTypes.BLOCK)
class PreTwoStem(object):
    """Class of two stems convolution.

    :param desc: description of PreTwoStem
    :type desc: Config
    """

    def __init__(self, desc):
        """Init PreTwoStem."""
        self._C = desc.C
        self.data_format = desc.data_format

    def __call__(self, x, training):
        """Forward function of PreTwoStem."""
        axis = 1 if self.data_format == 'channels_first' else 3
        out1 = tf.layers.conv2d(x, self._C // 2, 3, strides=2, padding='same',
                                use_bias=False, data_format=self.data_format)
        out1 = tf.layers.batch_normalization(out1, axis=axis, fused=True, training=training)
        out1 = tf.nn.relu(out1)
        out1 = tf.layers.conv2d(out1, self._C, 3, strides=2, padding='same',
                                use_bias=False, data_format=self.data_format)
        out1 = tf.layers.batch_normalization(out1, axis=axis, fused=True, training=training)
        out2 = tf.nn.relu(out1)
        out2 = tf.layers.conv2d(out2, self._C, 3, strides=2, padding='same',
                                use_bias=False, data_format=self.data_format)
        out2 = tf.layers.batch_normalization(out2, axis=axis, fused=True, training=training)
        return out1, out2
