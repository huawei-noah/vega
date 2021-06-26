# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The Graph Convolution Network model."""
import logging
import math
import tensorflow as tf
from vega.common import ClassType, ClassFactory


logger = logging.getLogger(__name__)


class GraphConvolution(object):
    """Graph Convolution Layer."""

    def __init__(self, in_features, out_features, bias=True, initializer=None, name='GC'):
        self.in_features = in_features
        self.out_features = out_features
        self.if_bias = bias
        self.name = name
        self.reset_parameters(initializer)

    def reset_parameters(self, initializer=None):
        """Reset parameters of layer."""
        stdv = 1. / math.sqrt(self.out_features)
        if initializer is None:
            initializer = tf.random_uniform_initializer(-stdv, stdv)
        self.weight = tf.get_variable(self.name + '/W', [self.in_features, self.out_features],
                                      initializer=initializer, trainable=True)
        self.bias = None
        if self.if_bias:
            self.bias = tf.get_variable(self.name + '/B', [self.out_features],
                                        initializer=initializer, trainable=True)

    def __call__(self, input, adj):
        """Forward function of graph convolution layer."""
        with tf.variable_scope(self.name):
            support = tf.matmul(input, self.weight)
            output = tf.matmul(adj, support)
            if self.bias is not None:
                return output + self.bias
            else:
                return output


@ClassFactory.register(ClassType.NETWORK)
class GCNRegressor(object):
    """Graph Convolution Network for regression."""

    def __init__(self, nfeat, ifsigmoid, layer_size=64):
        self.nfeat = nfeat
        self.ifsigmoid = ifsigmoid
        self.size = layer_size
        self.gc_initializer = tf.random_uniform_initializer(-0.05, 0.05)

    def _gc_bn_act(self, feat, adj, idx):
        nfeat = feat.get_shape().as_list()[-1]
        feat = GraphConvolution(nfeat, self.size, True, self.gc_initializer, 'GC_{}'.format(idx))(feat, adj)
        feat = tf.nn.relu(tf.layers.BatchNormalization()(tf.transpose(feat, [0, 2, 1])))
        feat = tf.transpose(feat, [0, 2, 1])
        return feat

    def __call__(self, input):
        """Forward function of GCN."""
        with tf.variable_scope('GCNRegressor'):
            shape = input.get_shape().as_list()
            adj, feat = tf.split(input, [shape[1], shape[2] - shape[1]], axis=2)
            n = 4
            for i in range(n):
                feat = self._gc_bn_act(feat, adj, i)
            feat_list = tf.split(feat, feat.shape[1], axis=1)
            embeddings = tf.squeeze(feat_list[-1], axis=1)
            y = tf.layers.Dense(1)(embeddings)
            y = tf.squeeze(y, axis=1)
            if self.ifsigmoid:
                return tf.math.sigmoid(y)
            else:
                return y
