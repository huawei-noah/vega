# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GCN Layers."""
import numpy as np
import tensorflow as tf


def first_approx(W, n):
    """1st-order approximation function.

    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    """
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    return np.mat(np.identity(n) + sinvD * A * sinvD)


def scaled_laplacian_tensor(W):
    """Create Normalized graph Laplacian function.

    :param W: tensor, [n_route, n_route], weighted adjacency matrix of G.
    :return: tensor, [n_route, n_route].
    """
    n = W.get_shape().as_list()[1]
    D = tf.reduce_sum(W, -1)
    L = tf.matrix_diag(D) - W

    sinvD = tf.matrix_diag(1 / tf.sqrt(D))
    L = sinvD @ L @ sinvD
    ret = L - tf.expand_dims(tf.eye(n), 0)
    ret = tf.where(tf.is_nan(ret), tf.zeros_like(ret), ret)
    return tf.squeeze(ret)


def cheb_poly_approx_tensor(L, ks, n=None):
    """Create Chebyshev polynomials approximation function.

    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    """
    n = int(L.get_shape()[1] if n is None else n)
    L0, L1 = tf.eye(n), L
    L_list = [L0, L1]
    for i in range(ks - 2):
        Ln = 2 * L * L - L0
        L0, L1 = L1, Ln
        L_list.append(Ln)
    return tf.concat(L_list, axis=-1)


def spatial_attention_layer(x):
    """Compute spatial attention.

    :param x: tensor, [batch_size,time_step,n_route,c_in]
    :return:
    """
    _, T, n, c = x.get_shape().as_list()
    W_1 = tf.get_variable(name='spatial_w_1', shape=[T, 1], dtype=tf.float32)
    W_2 = tf.get_variable(name='spatial_w_2', shape=[c, T], dtype=tf.float32)
    W_3 = tf.get_variable(name='spatial_w_3', shape=[c, 1], dtype=tf.float32)
    b_s = tf.get_variable(name='spatial_b_s', shape=[1, n, n], dtype=tf.float32)
    V_s = tf.get_variable(name='spatial_v_s', shape=[n, n], dtype=tf.float32)
    x_tmp = tf.transpose(x, [0, 2, 3, 1])
    lhs = tf.tensordot(tf.squeeze(tf.tensordot(x_tmp, W_1, axes=[[3], [0]]), axis=-1), W_2,
                       axes=[[2], [0]])
    rhs = tf.squeeze(tf.tensordot(x, W_3, axes=[[3], [0]]), axis=-1)
    s = tf.tensordot(tf.nn.sigmoid(tf.matmul(lhs, rhs) + b_s), V_s, axes=[[2], [0]])
    return tf.nn.softmax(s, axis=1)


def temporal_attention_layer(x):
    """Compute temporal attention.

    :param x: tensor, [batch_size,time_step,n_route,c_in]
    :return: tensor, [batch_size, time_step, time_step ]
    """
    _, T, n, c = x.get_shape().as_list()
    W_1 = tf.get_variable(name='temporal_w_1', shape=[n, 1], dtype=tf.float32)
    W_2 = tf.get_variable(name='temporal_w_2', shape=[c, n], dtype=tf.float32)
    W_3 = tf.get_variable(name='temporal_w_3', shape=[c, 1], dtype=tf.float32)
    b_s = tf.get_variable(name='temporal_b_s', shape=[1, T, T], dtype=tf.float32)
    V_s = tf.get_variable(name='temporal_v_s', shape=[T, T], dtype=tf.float32)
    x_tmp = tf.transpose(x, [0, 1, 3, 2])
    lhs = tf.tensordot(tf.squeeze(tf.tensordot(x_tmp, W_1, axes=[[3], [0]]), axis=-1), W_2,
                       axes=[[2], [0]])
    rhs = tf.transpose(tf.squeeze(tf.tensordot(x, W_3, axes=[[3], [0]]), axis=-1),
                       [0, 2, 1])
    s = tf.tensordot(tf.nn.sigmoid(tf.matmul(lhs, rhs) + b_s), V_s,
                     axes=[[2], [0]])
    s = tf.nn.softmax(s, axis=1)

    return s


def gconv_layer(x, Ks, number_of_layers, spatial_attention, c_in, c_out, adjacency_matrix=None):
    """Create Spectral-based graph convolution function.

    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of graph convolution.
    :param number_of_layers: int, number of gcn layers
    :param spatial_attention: boolean, flag to use spatial attention or not
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    :param adjacency_matrix: placeholder or None if not used.
    """
    _, T, n, _ = x.get_shape().as_list()

    kernels = tf.get_collection('graph_kernel')
    if adjacency_matrix is not None:
        L = scaled_laplacian_tensor(adjacency_matrix)
        Lk = cheb_poly_approx_tensor(L, Ks, n)
        kernels.append(Lk)

    for layer in range(number_of_layers):
        x_result = []
        x = tf.reshape(x, [-1, n, c_in])

        for i, kernel in enumerate(kernels):
            theta = tf.get_variable(
                name='theta_' + str(layer) + '_' + str(i),
                shape=[Ks * c_in, c_out],
                dtype=tf.float32)
            tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(theta))
            bs = tf.get_variable(
                name='bs_' + str(layer) + '_' + str(i),
                initializer=tf.zeros([c_out]),
                dtype=tf.float32)
            x = tf.transpose(x, [0, 2, 1])
            x_tmp = tf.reshape(x, [-1, n])

            if (adjacency_matrix is not None) and (i == len(kernels) - 1):
                x_mul = tf.matmul(x, kernel)
                x_mul = tf.reshape(x_mul, [-1, c_in, Ks, n])
            else:
                x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])

            x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
            x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out]) + bs

            x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])
            x_output = tf.nn.relu(x_gc[:, :, :, 0:c_out])

            x_result.append(x_output)

        x = tf.add_n(x_result)
        c_in = c_out

    return x


def GCN_GRU(x, Ks, channels, gru_layers, gcn_layers, keep_prob, temporal_attention, spatial_attention):
    """Create GCN_GRU block, which contains one spatial graph convolution layer follows by a gru layer.

    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param channels: list, channel configs of a single GCN_GRU block.
    :param gru_layers: int, number of GRU layers
    :param gcn_layers: int, number of GCN layers
    :param keep_prob: placeholder, prob of dropout.
    :param temporal_attention: boolean, whether to use temporal attention or not.
    :param spatial_attention: boolean, whether to use spatial attention or not.
    :param scope: str, variable scope.
    :param adjacency_matrix: placeholder, or None if not used.
    :return: tensor, [batch_size, 1, n_route, c_out].
    """
    c_input, c_gru, c_gcn = channels

    with tf.variable_scope('gcn_gru'):
        if temporal_attention:
            _, T, n, c = x.get_shape().as_list()
            s_t = temporal_attention_layer(x)
            x_tmp = tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [-1, n * c, T])
            x = tf.transpose(tf.reshape(tf.matmul(x_tmp, s_t), [-1, n, c, T]),
                             [0, 3, 1, 2])
    with tf.variable_scope('gcn_gru'):
        x_s = gconv_layer(x, Ks, gcn_layers, spatial_attention, 1, c_gcn)
        x_s = gru_layer(x_s, gru_layers, c_gru, keep_prob)
    return tf.nn.dropout(x_s, keep_prob)


def gru_layer(x, number_of_layers, c_out, keep_prob):
    """Create construct gru layer.

    :param x: tensor, [batch_size, time_step, n_route, c_in]
    :param number_of_layers: int, number of GRU layers
    :param c_out: int, output dimension
    :param keep_prob: placeholder, prob of dropout.
    :return: tensor, [batch_size, 1, n_route, c_out]
    """
    dim = x.get_shape().as_list()
    x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [-1, dim[1], dim[3]])

    cell = get_a_cell(c_out, keep_prob)
    if number_of_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(c_out, keep_prob) for _ in range(number_of_layers)])
    input_x = tf.unstack(x, num=dim[1], axis=1)
    outputs, last_states = tf.contrib.rnn.static_rnn(cell=cell, inputs=input_x, dtype=tf.float32)
    return tf.expand_dims(tf.reshape(outputs[-1], [-1, dim[2], c_out]), 1)


def get_a_cell(hidden_size, keep_prob):
    """Get helper function to construct a gru cell.

    :param hidden_size: int, dimension of the gru cell
    :param keep_prob: placeholder, drop out probability
    :return: tensor: rnn cell tensor
    """
    cell = tf.contrib.rnn.GRUCell(hidden_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return drop


def fully_con_layer(x, n, channel, scope):
    """Create Fully connected layer: maps multi-channels to one.

    :param x: tensor, [batch_size, 1, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param scope: str, variable scope.
    :return: tensor, [batch_size, 1, n_route, 1].
    """
    scope = str(scope)
    w = tf.get_variable(name='w_' + scope, shape=[1, 1, channel, 1], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.get_variable(name='b_' + scope, initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b


def output_layer_gru(x, scope):
    """Create fully connected layer layer: which map outputs of the last GCN_GRU block to a single-step prediction.

    :param x: tensor, [batch_size, 1, n_route, channel].
    :param scope: str, variable scope.
    :return: tensor, [batch_size, 1, n_route, 1].
    """
    _, _, n, channel = x.get_shape().as_list()
    return fully_con_layer(x, n, channel, scope)
