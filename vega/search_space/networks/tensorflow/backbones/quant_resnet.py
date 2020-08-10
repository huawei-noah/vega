# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ResNet models for pruning."""
import tensorflow as tf
from vega.search_space.networks import NetTypes, NetworkFactory
from ..blocks.resnet_block import _quant_basic_block
from vega.algorithms.compression.quant_ea.utils.tensorflow.quant_conv import QuantConv
import copy


@NetworkFactory.register(NetTypes.BACKBONE)
class QuantResNet(object):
    """PruneResNet.

    :param descript: network desc
    :type descript: dict
    """

    def __init__(self, descript):
        """Init QuantResNet."""
        self.net_desc = descript
        self.block_fn = _quant_basic_block
        self.num_blocks = descript.get('num_blocks', [3, 3, 3])
        self.num_classes = descript.get('num_classes', 10)
        self.nbit_w_list = descript.get('nbit_w_list', None)
        self.nbit_a_list = descript.get('nbit_a_list', None)
        self.in_planes = 16
        self.data_format = "channels_first"
        self.quant_info = dict()

    def _forward_quant_block(self, x, bottleneck, planes, num_blocks, stride, training, name, quant_info):
        """Create resolution block of ResNet."""
        idx = 0
        strides = [stride] + [1] * (num_blocks - 1)
        expansion = 4 if bottleneck else 1
        for stride in strides:
            x = self.block_fn(x, planes, training, self.data_format, name + '/block_' + str(idx),
                              strides=stride, quant_info=quant_info)
            self.in_planes = planes * expansion
            idx += 1
        return x

    def __call__(self, x, training, quant_info=None):
        """Forward function of ResNet."""
        if quant_info is not None:
            self.quant_info = quant_info
        if self.data_format == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])
        data_channel = x.get_shape()[1]
        if quant_info:
            self.nbit_w_list = copy.deepcopy(quant_info['nbit_w_list'])
            self.nbit_a_list = copy.deepcopy(quant_info['nbit_a_list'])

        if quant_info is not None and not quant_info['skip_1st_layer']:
            quant_conv = QuantConv(16, 3, name='/conv_1', strides=1,
                                   padding='same', groups=1, use_bias=False, data_format=self.data_format)
            quant_conv.quant_config(quant_info=quant_info, name='/conv_1')
            x = quant_conv(x)
        else:
            x = tf.layers.conv2d(x, 16, 3, padding='same', use_bias=False, data_format=self.data_format, name='conv_1')
        x = tf.layers.batch_normalization(x, axis=1 if self.data_format == 'channels_first' else 3,
                                          name='bn_1', training=training)
        x = tf.nn.relu(x)
        x = self._forward_quant_block(x, False, 16, self.num_blocks[0], stride=1,
                                      training=training, name='layer_1', quant_info=quant_info)
        x = self._forward_quant_block(x, False, 32, self.num_blocks[1], stride=2,
                                      training=training, name='layer_2', quant_info=quant_info)
        x = self._forward_quant_block(x, False, 64, self.num_blocks[2], stride=2,
                                      training=training, name='layer_3', quant_info=quant_info)
        x = tf.reduce_mean(x, [-2, -1], keepdims=True)
        out = tf.layers.dense(tf.reshape(x, [x.get_shape()[0], -1]), self.num_classes)
        return out
