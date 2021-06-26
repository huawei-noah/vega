# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is FasterRCNN network."""
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module


@ClassFactory.register(ClassType.NETWORK)
class GCN(Module):
    """Create ResNet Network."""

    def __init__(self, blocks=[[1, 32, 64]], kernel_size=4, gru_layers=1, gcn_layers=1, keep_prob=1,
                 temporal_attention=False, spatial_attention=False, adjacency_matrix=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.gru_layers = gru_layers
        self.gcn_layers = gcn_layers
        self.keep_prob = keep_prob
        self.temporal_attention = temporal_attention
        self.spatial_attention = spatial_attention
        self.adjacency_matrix = adjacency_matrix
        self.graph = 'spatial'
        self.use_gcn = True

    def call(self, inputs):
        """Override call function."""
        import tensorflow as tf
        from vega.networks.tensorflow.gcn.layers import GCN_GRU, output_layer_gru
        x, spatial_mx, temporal_mx = inputs[0], inputs[1][0], inputs[2][0]
        x = tf.cast(x, tf.float32)
        spatial_mx = tf.cast(spatial_mx, tf.float32)
        approx = self.update_with_approximation(spatial_mx, temporal_mx)
        tf.add_to_collection(name='graph_kernel', value=approx)
        for i, channels in enumerate(self.blocks):
            x = GCN_GRU(x, self.kernel_size, channels, self.gru_layers, self.gcn_layers, self.keep_prob,
                        self.temporal_attention, self.spatial_attention)
        return output_layer_gru(x, 'output_layer')

    def update_with_approximation(self, spatial_mx, temporal_mx):
        """Update with approximation.

        :param Ws: Spatial proximity adjacency matrix.
        :param Wt: Functional similarity adjacency matrix.
        :param W: Adjacency matrix (in the other cases: spatial OR temporal)
        :param n_route: Number of base stations.
        """
        from vega.networks.tensorflow.gcn.layers import scaled_laplacian_tensor, cheb_poly_approx_tensor
        laplacian = scaled_laplacian_tensor(spatial_mx)
        approx = cheb_poly_approx_tensor(laplacian, self.kernel_size)
        return approx
