# -*- coding:utf-8 -*-

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

"""This is FasterRCNN network."""
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module


@ClassFactory.register(ClassType.NETWORK)
class GCN(Module):
    """Create ResNet Network."""

    def __init__(self, blocks=None, kernel_size=4, gru_layers=1, gcn_layers=1, keep_prob=1,
                 temporal_attention=False, spatial_attention=False, adjacency_matrix=None):
        super().__init__()
        if blocks is None:
            blocks = [[1, 32, 64], ]
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
