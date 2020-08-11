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
from ..blocks.resnet_block import _prune_basic_block


@NetworkFactory.register(NetTypes.BACKBONE)
class PruneResNet(object):
    """PruneResNet.

    :param descript: network desc
    :type descript: dict
    """

    def __init__(self, descript):
        """Init PruneResNet."""
        self.net_desc = descript
        self.block = _prune_basic_block
        self.encoding = descript.get('encoding')
        self.chn = descript.get('chn')
        self.chn_node = descript.get('chn_node')
        self.chn_mask = descript.get('chn_mask', None)
        self.chn_node_mask = descript.get('chn_node_mask', None)
        self.num_blocks = descript.get('num_blocks', [3, 3, 3])
        self.num_classes = descript.get('num_classes', 10)
        self.in_planes = self.chn_node[0]
        self.data_format = "channels_first"
        self.scope_name = 'PruneResnet'

    def _forward_prune_block(self, x, bottleneck, block, planes, inner_planes, num_blocks, stride, training, name):
        """Create resolution block of ResNet."""
        idx = 0
        strides = [stride] + [1] * (num_blocks - 1)
        expansion = 4 if bottleneck else 1
        for stride in strides:
            x = block(x, planes, inner_planes[idx], training, self.data_format,
                      name + '/block_' + str(idx), strides=stride)
            self.in_planes = planes * expansion
            idx += 1
        return x

    def __call__(self, x, training):
        """Forward function of ResNet."""
        if self.data_format == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.layers.conv2d(x, self.chn_node[0], 3, padding='same', use_bias=False,
                             data_format=self.data_format, name='conv_1')
        x = tf.layers.batch_normalization(x, axis=1 if self.data_format == 'channels_first' else 3,
                                          name='bn_1', training=training)
        x = self._forward_prune_block(x, False, self.block, self.chn_node[1], self.chn[0:3],
                                      self.num_blocks[0], stride=1, training=training, name='layer_1')
        x = self._forward_prune_block(x, False, self.block, self.chn_node[2], self.chn[3:6],
                                      self.num_blocks[1], stride=2, training=training, name='layer_2')
        x = self._forward_prune_block(x, False, self.block, self.chn_node[3], self.chn[6:9],
                                      self.num_blocks[2], stride=2, training=training, name='layer_3')
        x = tf.nn.relu(x)
        x = tf.reduce_mean(x, [-2, -1], keepdims=True)
        out = tf.layers.dense(tf.reshape(x, [x.get_shape()[0], -1]), self.num_classes)
        return out
