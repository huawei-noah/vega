# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Backbone of mobilenet v2."""
import tensorflow as tf


class MobileNetV2Backbone(object):
    """Backbone of mobilenet v2."""

    def __init__(self, load_path=None, data_format='channels_first'):
        """Construct MobileNetV2 class.

        :param load_path: path for saved model
        """
        self.data_format = data_format
        self.channel_axis = 1 if data_format == 'channel_first' else 3
        self.load_path = load_path

    def block(self, x, input_filters, output_filters, expansion, stride, training):
        """Mobilenetv2 block."""
        shortcut = x
        x = tf.layers.conv2d(x, input_filters * expansion, kernel_size=1, use_bias=False,
                             padding='same', data_format=self.data_format)
        x = tf.layers.batch_normalization(x, axis=self.channel_axis, training=training)
        x = tf.nn.relu6(x)
        x = tf.layers.separable_conv2d(x, input_filters * expansion, kernel_size=3, strides=stride, use_bias=False,
                                       padding='same', data_format=self.data_format)
        x = tf.layers.batch_normalization(x, axis=self.channel_axis, training=training)
        x = tf.nn.relu6(x)
        x = tf.layers.conv2d(x, output_filters, kernel_size=1, use_bias=False,
                             padding='same', data_format=self.data_format)
        x = tf.layers.batch_normalization(x, axis=self.channel_axis, training=training)
        if stride == 2:
            return x
        else:
            if input_filters != output_filters:
                shortcut = tf.layers.conv2d(shortcut, output_filters, kernel_size=1, use_bias=False,
                                            padding='same', data_format=self.data_format)
            return shortcut + x

    def blocks(self, x, expansion, output_filters, repeat, stride, training):
        """Mobilenetv2 blocks."""
        input_filters = int(x.get_shape()[self.channel_axis])

        # first layer should take stride into account
        x = self.block(x, input_filters, output_filters, expansion, stride, training)

        for _ in range(1, repeat):
            x = self.block(x, input_filters, output_filters, expansion, 1, training)

        return x

    def __call__(self, x, training):
        """Do an inference on MobileNetV2.

        :param x: input tensor
        :return: output tensor
        """
        outs = []
        net = tf.layers.conv2d(x, 32, 3, strides=2, use_bias=False, padding='same', data_format=self.data_format)
        x = tf.layers.batch_normalization(x, axis=self.channel_axis, training=training)
        x = tf.nn.relu6(x)

        expansion_list = [1] + [6] * 6
        output_filter_list = [16, 24, 32, 64, 96, 160, 320]
        repeat_list = [1, 2, 3, 4, 3, 3, 1]
        stride_list = [1, 2, 2, 2, 1, 2, 1]

        for i in range(7):
            x = self.blocks(x, expansion_list[i], output_filter_list[i], repeat_list[i], stride_list[i], training)
            if i in [1, 2, 4, 6]:
                outs.append(x)

        return outs
