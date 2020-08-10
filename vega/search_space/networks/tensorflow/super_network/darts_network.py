# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Darts Network."""
import tensorflow as tf
from vega.core.common import Config
from vega.search_space.networks import NetTypes, NetTypesMap, NetworkFactory


@NetworkFactory.register(NetTypes.SUPER_NETWORK)
class DartsNetwork(object):
    """Base Darts Network of classification.

    :param desc: darts description
    :type desc: Config
    """

    def __init__(self, desc):
        """Init DartsNetwork."""
        super(DartsNetwork, self).__init__()
        self.desc = desc
        self.network = desc.network
        self._C = desc.init_channels
        self._classes = desc.num_classes
        self.input_size = desc.input_size
        self.data_format = desc.get('data_format', 'channels_first')
        self._auxiliary = desc.auxiliary
        self.search = desc.search
        self.drop_path_prob = desc.get('drop_path_prob', 0.)
        self.fp16 = tf.float16 if desc.get('fp16', False) else tf.float32
        self.scope_name = 'DartsNetwork'
        if self._auxiliary:
            self._aux_size = desc.aux_size
            self._auxiliary_layer = desc.auxiliary_layer

    def build_network(self):
        """Build Darts Network."""
        C_curr = self._network_stems(self.network[0])
        C_prev, C_aux = self._network_cells(self.network[1:], C_curr)
        if not self.search and self._auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_aux, self._classes,
                                                self._aux_size, self.data_format)
        self.classifier = tf.layers.dense
        if self.search:
            self._initialize_alphas()

    def _network_stems(self, stem):
        """Build stems part.

        :param stem: stem part of network
        :type stem: torch.nn.Module
        :return: stem's output channel
        :rtype: int
        """
        stem_desc = {'C': self._C, 'stem_multi': 3, 'data_format': self.data_format}
        stem_class = NetworkFactory.get_network(NetTypesMap['block'], stem)
        self.stem = stem_class(Config(stem_desc))
        return self.stem.C_curr

    def _network_cells(self, network_list, C_curr):
        """Build cells part.

        :param network_list: list of cell's name
        :type network_list: list of str
        :param C_curr: input channel of cells
        :type C_curr: int
        """
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self._C
        self.cells = []
        reduction_prev = True if C_curr == C_prev else False
        for i, name in enumerate(network_list):
            if name == 'reduce':
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = self.build_cell(name, C_prev_prev, C_prev, C_curr,
                                   reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            multiplier = len(self.desc[name]['concat'])
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            if self._auxiliary and i == self._auxiliary_layer:
                C_aux = C_prev
        return C_prev, C_aux if self._auxiliary else None

    def build_cell(self, name, C_prev_prev, C_prev, C_curr, reduction, reduction_prev):
        """Build cell for Darts Network.

        :param name: cell name
        :type name: str
        :param C_prev_prev: channel of previous of previous cell
        :type C_prev_prev: int
        :param C_prev: channel of previous cell
        :type C_prev: int
        :param C_curr: channel of current cell
        :type C_curr: int
        :param reduction: whether to reduce resolution in this cell
        :type reduction: bool
        :param reduction_prev: whether to reduce resolution in previous cell
        :return: object of cell
        :rtype: class type of cell
        """
        cell_desc = {
            'genotype': self.desc[name]['genotype'],
            'steps': self.desc[name]['steps'],
            'concat': self.desc[name]['concat'],
            'C_prev_prev': C_prev_prev,
            'C_prev': C_prev,
            'C': C_curr,
            'reduction': reduction,
            'reduction_prev': reduction_prev,
            'search': self.search,
            'data_format': self.data_format
        }
        cell_type = self.desc[name]['type']
        cell_name = self.desc[name]['name']
        cell_class = NetworkFactory.get_network(
            NetTypesMap[cell_type], cell_name)
        return cell_class(Config(cell_desc))

    def _initialize_alphas(self):
        """Initialize architecture parameters."""
        k = len(self.desc.normal.genotype)
        num_ops = len(self.desc.normal.genotype[0][0])
        self.alphas_normal = tf.get_variable('alphas_normal', initializer=1e-3 * tf.random.normal((k, num_ops)))
        self.alphas_reduce = tf.get_variable('alphas_reduce', initializer=1e-3 * tf.random.normal((k, num_ops)))

    def get_arch_ops(self):
        """Get arch ops."""
        all_weight = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        arch_ops = [t for t in all_weight if t.name.startswith(self.scope_name + '/alphas')]
        return arch_ops

    def get_weight_ops(self):
        """Get weight ops."""
        all_weight = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weight_ops = [t for t in all_weight if not t.name.startswith(self.scope_name + '/alphas')]
        return weight_ops

    @property
    def arch_weights(self):
        """Get arch weights of numpy list type."""
        alphas_normal = tf.get_default_graph().get_tensor_by_name('DartsNetwork/alphas_normal:0')
        alphas_reduce = tf.get_default_graph().get_tensor_by_name('DartsNetwork/alphas_reduce:0')
        alphas_normal = tf.nn.softmax(alphas_normal, axis=-1)
        alphas_reduce = tf.nn.softmax(alphas_reduce, axis=-1)
        return [alphas_normal, alphas_reduce]

    def __call__(self, input, training):
        """Forward function of Darts Network."""
        stem_training = training
        if self.search:
            # during search, bn layers must be in training mode, otherwise validation will be random guess
            training = True
        if self.data_format == 'channels_first':
            input = tf.transpose(input, [0, 3, 1, 2])

        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE) as scope:
            self.build_network()
            s0, s1 = self.stem(input, training=stem_training)
            s0 = tf.identity(s0, 'stem_s0')
            s1 = tf.identity(s1, 'stem_s1')
            for i, cell in enumerate(self.cells):
                if self.search:
                    if self.desc.network[i + 1] == 'reduce':
                        weights = tf.nn.softmax(self.alphas_reduce, axis=-1)
                    else:
                        weights = tf.nn.softmax(self.alphas_normal, axis=-1)
                else:
                    weights = None
                s0, s1 = s1, cell(s0, s1, training, weights, drop_prob=self.drop_path_prob)
                s1 = tf.identity(s1, 'cell_{}'.format(i))
                if not self.search:
                    if self._auxiliary and i == self._auxiliary_layer:
                        logits_aux = self.auxiliary_head(s1, training=training)
            out = tf.reduce_mean(s1, [-2, -1], keepdims=True)
            out = tf.reshape(out, [out.get_shape()[0], -1])
            logits = self.classifier(out, units=self._classes)
            logits = tf.identity(logits, 'final_dense')
            if self._auxiliary and not self.search:
                return logits, logits_aux
            else:
                return logits


class AuxiliaryHead(object):
    """Auxiliary Head of Network.

    :param C: input channels
    :type C: int
    :param num_classes: numbers of classes
    :type num_classes: int
    :param input_size: input size
    :type input_size: int
    """

    def __init__(self, C, num_classes, input_size, data_format):
        """Init AuxiliaryHead."""
        self.data_format = data_format
        self.num_classes = num_classes
        self.s = input_size - 5
        self.classifier = tf.layers.dense

    def __call__(self, x, training):
        """Forward function of Auxiliary Head."""
        axis = 1 if self.data_format == 'channels_first' else 3
        x = tf.nn.relu(x)
        x = tf.layers.average_pooling2d(x, 5, strides=self.s, padding='same', data_format=self.data_format)
        x = tf.layers.conv2d(x, 128, 1, use_bias=False, data_format=self.data_format)
        x = tf.layers.batch_normalization(x, axis=axis, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, 768, 2, use_bias=False, data_format=self.data_format)
        x = tf.layers.batch_normalization(x, axis=axis, training=training)
        x = tf.nn.relu(x)
        x = self.classifier(tf.reshape(x, [x.get_shape()[0], -1]), self.num_classes)
        return x
