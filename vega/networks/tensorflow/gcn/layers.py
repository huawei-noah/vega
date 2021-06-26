# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
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
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)
    # return np.mat(sinvD * A * sinvD)


def scaled_laplacian_tensor(W):
    """Create Normalized graph Laplacian function.

    :param W: tensor, [n_route, n_route], weighted adjacency matrix of G.
    :return: tensor, [n_route, n_route].
    """
    # d ->  diagonal degree matrix
    n = W.get_shape().as_list()[1]
    D = tf.reduce_sum(W, -1)

    # L -> graph Laplacian
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
    # d = tf.expand_dims(tf.eye(n), 0)
    # d = tf.eye(n)
    L0, L1 = tf.eye(n), L
    L_list = [L0, L1]
    for i in range(ks - 2):
        # L0 = d + L - L
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
    # x -> [batch_size, n_route, c_in, time_step]
    x_tmp = tf.transpose(x, [0, 2, 3, 1])
    lhs = tf.tensordot(tf.squeeze(tf.tensordot(x_tmp, W_1, axes=[[3], [0]]), axis=-1), W_2,
                       axes=[[2], [0]])  # [batch_size, n_route, time_step]
    # x : [batch_size, time_step, n_route, c_in]
    rhs = tf.squeeze(tf.tensordot(x, W_3, axes=[[3], [0]]), axis=-1)  # [batch_size, time_step, n_route]
    # s = tf.matmul(tf.nn.sigmoid(tf.matmul(lhs, rhs) + b_s), V_s) # [batch_size, n_route, n_route]
    s = tf.tensordot(tf.nn.sigmoid(tf.matmul(lhs, rhs) + b_s), V_s, axes=[[2], [0]])  # [batch_size, n_route, n_route]
    # normalization
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
    # x -> [batch_size, time_step, c_in, n_route]
    x_tmp = tf.transpose(x, [0, 1, 3, 2])
    lhs = tf.tensordot(tf.squeeze(tf.tensordot(x_tmp, W_1, axes=[[3], [0]]), axis=-1), W_2,
                       axes=[[2], [0]])  # [batch_size, time_step, n_route]
    rhs = tf.transpose(tf.squeeze(tf.tensordot(x, W_3, axes=[[3], [0]]), axis=-1),
                       [0, 2, 1])  # [batch_size, n_route, time_step]
    s = tf.tensordot(tf.nn.sigmoid(tf.matmul(lhs, rhs) + b_s), V_s,
                     axes=[[2], [0]])  # [batch_size, time_step, time_step]
    # normalization
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

    # Currently only support for one kernel when using variable adjacency matrix
    kernels = tf.get_collection('graph_kernel')  # list of graph kernels defined in main.py
    if adjacency_matrix is not None:
        L = scaled_laplacian_tensor(adjacency_matrix)
        Lk = cheb_poly_approx_tensor(L, Ks, n)
        kernels.append(Lk)

    for layer in range(number_of_layers):
        x_result = []
        x = tf.reshape(x, [-1, n, c_in])  # [batch_size*time_step, n_route, c_in]

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

            # kernel: graph kernel: tensor, [n_route, Ks*n_route]
            # n = tf.shape(kernel)[0]
            # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
            x = tf.transpose(x, [0, 2, 1])
            x_tmp = tf.reshape(x, [-1, n])

            if (adjacency_matrix is not None) and (i == len(kernels) - 1):  # dynamic graph
                # x_tmp ->  [batch_size*c_in, n_route]
                # adjacency_matrix -> [batch_size, n_route, Ks*n_route]
                # x_mul = tf.tensordot(
                #   x, adjacency_matrix, axes=[[2], [1]]) # [batch_size, c_in, batch_size, Ks*n_route]

                # for each x,matrix pair in batch_size:
                # [batch_Size,c_in, n_route] * [batch_Size, n_route, Ks*n_route] -> [c_in, Ks*n_route]
                # [batch_Size, c_in, Ks*n_route]
                x_mul = tf.matmul(x, kernel)
                x_mul = tf.reshape(x_mul, [-1, c_in, Ks, n])
            else:
                # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
                x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])

            # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
            x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
            # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
            x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out]) + bs

            x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])
            x_output = tf.nn.relu(x_gc[:, :, :, 0:c_out])  # activation

            x_result.append(x_output)

        x = tf.add_n(x_result)  # add the result from multi-graph
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
            s_t = temporal_attention_layer(x)  # [batch_size, time_step, time_step ]
            x_tmp = tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [-1, n * c, T])  # [batch_size, n_route*c_out, time_step]
            x = tf.transpose(tf.reshape(tf.matmul(x_tmp, s_t), [-1, n, c, T]),
                             [0, 3, 1, 2])  # [batch_size, time_step,n_route,c_out]
    # first GCN then GRU
    with tf.variable_scope('gcn_gru'):
        x_s = gconv_layer(x, Ks, gcn_layers, spatial_attention, 1, c_gcn)  # [batch_size, time_step, n_route, c_out]
        x_s = gru_layer(x_s, gru_layers, c_gru, keep_prob)
    # x_ln = layer_norm(x_t, 'layer_norm_'+scope)
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
    x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [-1, dim[1], dim[3]])  # [batch_size*n_route, time_step, c_in]

    cell = get_a_cell(c_out, keep_prob)
    if number_of_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(c_out, keep_prob) for _ in range(number_of_layers)])

    # _, last_states = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32) #last_states: [batch_size*n_route,c_out]
    input_x = tf.unstack(x, num=dim[1], axis=1)  # needed if use tf.contrib.rnn.static_rnn
    outputs, last_states = tf.contrib.rnn.static_rnn(cell=cell, inputs=input_x, dtype=tf.float32)
    # print(outputs[-1],last_states[-1]) # these two are the same
    return tf.expand_dims(tf.reshape(outputs[-1], [-1, dim[2], c_out]), 1)  # [batch_size, 1, n_route, c_out]


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
