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

"""The DeepFM model."""
import copy
import torch
from vega.common import ClassType, ClassFactory
from .fis.layers import LinearLayer, EmbeddingLayer, \
    FactorizationMachineLayer, MultiLayerPerceptron


@ClassFactory.register(ClassType.NETWORK)
class DeepFactorizationMachineModel(torch.nn.Module):
    """DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. https://arxiv.org/abs/1703.04247.

    :param input_dim: feature space of dataset
    :type input_dim: int
    :param input_dim4lookup: feature number in `feature_id`, usually equals to number of non-zero features
    :type input_dim4lookup: int
    :param embed_dim: length of each feature's latent vector(embedding vector)
    :type embed_dim: int
    :param hidden_dims: width of each hidden layer, from bottom to top
    :type hidden_dims: list of int
    :param dropout_prob: dropout probability of all hidden layer
    :type dropout_prob: float
    :param batch_norm: applies batch normalization before activation, defaults to False
    :type batch_norm: bool, optional
    :param layer_norm: applies layer normalization before activation, defaults to False
    :type layer_norm: bool, optional
    """

    def __init__(self, net_desc):
        """
        Construct the DeepFactorizationMachineModel class.

        :param net_desc: config of the structure
        """
        super().__init__()
        self.desc = copy.deepcopy(net_desc)

        self.linear = LinearLayer(net_desc['input_dim'])
        self.embedding = EmbeddingLayer(net_desc['input_dim'], net_desc['embed_dim'])
        self.fm = FactorizationMachineLayer()
        self.mlp_input_dim = net_desc['input_dim4lookup'] * net_desc['embed_dim']
        self.mlp = MultiLayerPerceptron(
            self.mlp_input_dim, net_desc['hidden_dims'], net_desc['dropout_prob'],
            batch_norm=net_desc['batch_norm'], layer_norm=net_desc['layer_norm'])
        self.l1_cover_params = []
        self.l2_cover_params = []

    def forward(self, feature_id):
        """Calculate logits of pctr for given batch of samples.

        :param feature_id: a batch of feature id, tensor of size ``(batch_size, input_dim4lookup)``
        :type feature_id: torch.int
        :return: logits of pctr for given batch of samples
        :rtype: tensor.float, size ``(batch_size, 1)``
        """
        feature_val = None
        linear_score = self.linear(feature_id, feature_val).squeeze(1)
        embed_v = self.embedding(feature_id, feature_val)
        fm_score = self.fm(embed_v).squeeze(1)
        mlp_score = self.mlp(embed_v.view(-1, self.mlp_input_dim)).squeeze(1)
        return linear_score + fm_score + mlp_score
