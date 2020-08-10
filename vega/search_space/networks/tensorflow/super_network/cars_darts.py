# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""SuperNet for CARS-DARTS."""
import logging
import tensorflow as tf
from vega.search_space.networks import NetworkFactory, NetTypes
from vega.search_space.networks.tensorflow.super_network import DartsNetwork
import numpy as np
import copy

logger = logging.getLogger(__name__)


@NetworkFactory.register(NetTypes.SUPER_NETWORK)
class CARSDartsNetwork(DartsNetwork):
    """Base CARS-Darts Network of classification.

    :param desc: darts description
    :type desc: Config
    """

    def __init__(self, desc, scope_name='CarsNetwork'):
        """Init CARSDartsNetwork."""
        super(CARSDartsNetwork, self).__init__(desc)
        self.scope_name = scope_name
        self.steps = self.desc.normal.steps
        self.num_ops = self.num_ops()
        self.len_alpha = self.len_alpha()

    def len_alpha(self):
        """Get number of path."""
        k_normal = len(self.desc.normal.genotype)
        return k_normal

    def num_ops(self):
        """Get number of candidate operations."""
        num_ops = len(self.desc.normal.genotype[0][0])
        return num_ops

    def __call__(self, input, alpha=None, training=True):
        """Forward a model that specified by alpha.

        :param input: An input tensor
        :type input: Tensor
        """
        stem_training = training
        if self.data_format == 'channels_first':
            input = tf.transpose(input, [0, 3, 1, 2])

        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE) as scope:
            self.build_network()
            if self.search:
                training = True
                alphas_normal = alpha[:self.len_alpha]
                alphas_reduce = alpha[self.len_alpha:]
            s0, s1 = self.stem(input, training=stem_training)
            for i, cell in enumerate(self.cells):
                if self.search:
                    if self.desc.network[i + 1] == 'reduce':
                        weights = alphas_reduce
                    else:
                        weights = alphas_normal
                else:
                    weights = None
                s0, s1 = s1, cell(s0, s1, training, weights, drop_prob=self.drop_path_prob)
                if not self.search:
                    if self._auxiliary and i == self._auxiliary_layer:
                        logits_aux = self.auxiliary_head(s1, training=training)
            out = tf.reduce_mean(s1, [-2, -1], keepdims=True)
            out = tf.reshape(out, [out.get_shape()[0], -1])
            logits = self.classifier(out, units=self._classes)
            if self._auxiliary and not self.search:
                return logits, logits_aux
            else:
                return logits
