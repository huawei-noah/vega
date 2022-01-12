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

"""
Common modules in ctr prediction task.

Input features are usually sparse, so it benefits if they are represented in a sparse way.

For Example, feature `x = [0,0,0,1,0.5,0,0,0,0,0]` can be represented using
arrays `feature_id = [3, 4]` and `feature_val = [1, 0.5]`. The former one
denotes indexes of non-zero features and the latter one denotes corresponding values.

In many settings, `x` is binary, which means `feature_val` is always a ones array,
which makes it optional.
"""

from itertools import combinations
import logging
import torch
import torch.nn as nn


def generate_pair_index(n, order=2, selected_pairs=None):
    """Return enumeration of feature combination pair index.

    :param n: number of valid features, usually equals to `input_dim4lookup`
    :type n: int
    :param order: order of interaction. defaults to 2
    :type order: int
    :param selected_pairs: specifying selected pair of index
    :type selected_pairs: sequence of tuples, optional
    :return: a list of tuple, each containing feature index
    :rtype: list of tuple

    :Example:

    >>> generate_pair_index(5, 2)
    >>> [(0, 0, 0, 0, 1, 1, 1, 2, 2, 3),
         (1, 2, 3, 4, 2, 3, 4, 3, 4, 4)]
    >>> generate_pair_index(5, 3)
    >>> [(0, 0, 0, 0, 0, 0, 1, 1, 1, 2),
         (1, 1, 1, 2, 2, 3, 2, 2, 3, 3),
         (2, 3, 4, 3, 4, 4, 3, 4, 4, 4)]
    >>> generate_pair_index(5, 2, [(0,1),(1,3),(2,3)])
    >>> [(0, 1, 2), (1, 3, 3)]
    """
    if n < 2:
        raise ValueError("undefined. please ensure n >= 2")
    pairs = list(combinations(range(n), order))
    if selected_pairs is not None and len(selected_pairs) > 0:
        valid_pairs = set(selected_pairs)
        pairs = list(filter(lambda x: x in valid_pairs, pairs))
        logging.info("Using following selected feature pairs \n{}".format(pairs))
        if len(pairs) != len(selected_pairs):
            logging.warning("Pair number {} != specified pair number {}".format(len(pairs), len(selected_pairs)))
    return list(zip(*pairs))


class LinearLayer(torch.nn.Module):
    """Logistic Regression module."""

    def __init__(self, input_dim):
        """Class of LinearLayer.

        :param input_dim: feature space of dataset
        :type input_dim: int
        """
        super(LinearLayer, self).__init__()
        self.w = torch.nn.Embedding(input_dim, 1)
        torch.nn.init.xavier_uniform_(self.w.weight.data)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, feature_id, feature_val=None):
        """Logit = W^T*X + bias.

        :param feature_id: a batch of feature id, tensor of size ``(batch_size, input_dim4lookup)``
        :type feature_id: torch.int
        :param feature_val: a batch of feature value, defaults to None
        :type feature_val: torch.float, optional
        :return: logit of LR
        :rtype: torch.float
        """
        if feature_val is None:
            return torch.sum(self.w(feature_id), dim=1) + self.bias
        return torch.sum(self.w(feature_id).squeeze(2) * feature_val, dim=1) + self.bias


class EmbeddingLayer(torch.nn.Module):
    """Embedding module.

    It is a sparse to dense operation that lookup embedding for given features.
    """

    def __init__(self, input_dim, embed_dim):
        """Class of EmbeddingLayer.

        :param input_dim: feature space of dataset
        :type input_dim: int
        :param embed_dim: length of each feature's latent vector aka embedding vector
        :type embed_dim: int
        """
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(input_dim, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, feature_id, feature_val=None):
        """Forward function.

        :param feature_id: a batch of feature id, tensor of size ``(batch_size, input_dim4lookup)``
        :type feature_id: torch.int
        :param feature_val: a batch of feature value, defaults to None
        :type feature_val: torch.float, optional
        :return: embedding tensor of size ``(batch_size, input_dim4lookup, embed_dim)``
        :rtype: torch.float
        """
        if feature_val is None:
            return self.embedding(feature_id)
        return self.embedding(feature_id) * feature_val.unsqueeze(-1)


class FactorizationMachineLayer(torch.nn.Module):
    """Factorization Machines module.

    :param reduce_sum: whether to sum interaction score of all feature pairs, defaults to `True`
    :type reduce_sum: bool, optional
    """

    def __init__(self, reduce_sum=True):
        super(FactorizationMachineLayer, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, embed_matrix):
        """Y = sum {<emebd_i, embed_j>}.

        :param embed_matrix: a batch of embedding features, tensor of size ``(batch_size, input_dim4lookup, embed_dim)``
        :type embed_matrix: torch.float
        :return: FM layer's score.
        :rtype: torch.float, size ``(batch_size, 1)``(`reduce_sum==True`)
                or size ``(batch_size, embed_dim)``(`reduce_sum==False`)
        """
        square_of_sum = torch.sum(embed_matrix, dim=1) ** 2
        sum_of_square = torch.sum(embed_matrix ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class NormalizedWeightedFMLayer(torch.nn.Module):
    """NormalizedWeightedFMLayer module for autogate."""

    def __init__(self, input_dim4lookup, alpha_init_mean=0.5, alpha_init_radius=0.001,
                 alpha_activation='tanh', selected_pairs=None,
                 reduce_sum=True):
        """
        Autogate key component, learning to identify & select useful feature interactions with the help of `alpha`.

        :param input_dim4lookup: feature number in `feature_id`, usually equals to number of non-zero features.
        :type input_dim4lookup: int
        :param alpha_init_mean: mean of initialization value for `alpha`, defaults to 0.5
        :type alpha_init_mean: float, optional
        :param alpha_init_radius: radius of initialization range for `alpha`, defaults to 0.001
        :type alpha_init_radius: float, optional
        :param alpha_activation: activation function for `alpha`, one of 'tanh' or 'identity', defaults to 'tanh'
        :type alpha_activation: str, optional
        :param selected_pairs: use selected feature pairs (denoted by their index in given arrangement), defaults to []
        :type selected_pairs: list of tuple, optional
        :param reduce_sum: whether to sum interaction score of feature pairs, defaults to `True`
        :type reduce_sum: bool, optional
        """
        super(NormalizedWeightedFMLayer, self).__init__()
        self.reduce_sum = reduce_sum
        self.register_buffer('pair_indexes', torch.tensor(generate_pair_index(input_dim4lookup, 2, selected_pairs)))
        interaction_pair_number = len(self.pair_indexes[0])
        self._alpha = torch.nn.Parameter(
            torch.empty(interaction_pair_number).uniform_(
                alpha_init_mean - alpha_init_radius,
                alpha_init_mean + alpha_init_radius),
            requires_grad=True)
        self.activate = nn.Tanh() if alpha_activation == 'tanh' else nn.Identity()
        logging.info("using activation {}".format(self.activate))
        self.batch_norm = torch.nn.BatchNorm1d(interaction_pair_number, affine=False, momentum=0.01, eps=1e-3)

    def forward(self, embed_matrix):
        """Y = sum{alpha_i_j * BatchNorm(<e_i, e_j>)}.

        :param embed_matrix: a batch of embedding features, tensor of size ``(batch_size, input_dim4lookup, embed_dim)``
        :type embed_matrix: torch.float
        :return: normalized weighted FM layer's score
        :rtype: torch.float, size ``(batch_size, 1)``(`reduce_sum==True`) or
            size ``(batch_size, embed_dim)``(`reduce_sum==False`)
        """
        feat_i, feat_j = self.pair_indexes
        embed_i = torch.index_select(embed_matrix, 1, feat_i)
        embed_j = torch.index_select(embed_matrix, 1, feat_j)
        embed_product = torch.sum(torch.mul(embed_i, embed_j), dim=2)
        normed_emded_product = self.batch_norm(embed_product)
        weighted_embed_product = torch.mul(normed_emded_product, self.activate(self._alpha.unsqueeze(0)))
        if self.reduce_sum:
            return torch.sum(weighted_embed_product, dim=1, keepdim=True)
        return weighted_embed_product


class MultiLayerPerceptron(torch.nn.Module):
    """MultiLayerPerceptron module."""

    def __init__(self, input_dim, hidden_dims, dropout_prob, add_output_layer=True, batch_norm=False, layer_norm=False):
        """
        Multi Layer Perceptron module.

        :param input_dim: feature space of dataset
        :type input_dim: int
        :param hidden_dims: width of each hidden layer, from bottom to top
        :type hidden_dims: list of int
        :param dropout_prob: dropout probability of all hidden layer
        :type dropout_prob: float
        :param add_output_layer: whether to add an output layer for binary classification, defaults to `True`
        :type add_output_layer: bool, optional
        :param batch_norm: applies batch normalization before activation, defaults to `False`
        :type batch_norm: bool, optional
        :param layer_norm: applies layer normalization before activation, defaults to `False`
        :type layer_norm: bool, optional
        """
        if batch_norm and layer_norm:
            logging.warning("batch norm and layer norm are not supposed to work together! be careful...")
        super(MultiLayerPerceptron, self).__init__()
        layers = list()
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim, bias=True))  # default init: uniform
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_dim))
            if layer_norm:
                layers.append(torch.nn.LayerNorm(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout_prob))
            input_dim = hidden_dim
        if add_output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, embed_matrix):
        """Forward function.

        :param embed_matrix: a batch of embedding features, tensor of size ``(batch_size, input_dim4lookup, embed_dim)``
        :type embed_matrix: torch.float
        :return: MLP module's score
        :rtype: torch.float, size ``(batch_size, 1)``(`add_output_layer==True`) or
            size ``(batch_size, hidden_dims[-1])``(`add_output_layer==False`)
        """
        return self.mlp(embed_matrix)


class FeatureGroupLayer(torch.nn.Module):
    """FeatureGroupLayer module."""

    def __init__(self, input_dim4lookup, embed_dim, bucket_num, temperature,
                 lambda_c, epsilon=1e-20):
        """
        Autogroup key component, applies differentiable feature group selection.

        :param input_dim4lookup: feature number in `feature_id`, usually equals to number of non-zero features
        :type input_dim4lookup: int
        :param embed_dim: length of each feature's latent vector(embedding vector)
        :type embed_dim: int
        :param bucket_num: number of hash buckets
        :type bucket_num: int
        :param temperature: temperature in Gumbel-Softmax
        :type temperature: float (0,1]
        :param lambda_c: compensation coefficient for feature interaction score
        :type lambda_c: float [0,1]
        :param epsilon: term added to the denominator to improve numerical stabilit, defaults to 1e-20
        :type epsilon: float, optional
        """
        super(FeatureGroupLayer, self).__init__()
        # params for gumbel-softmax sampling
        self.structure_logits = torch.nn.Parameter(
            torch.empty(input_dim4lookup, bucket_num, 2).uniform_(-0.001, 0.001),
            requires_grad=True)  # (input_dim4lookup, bucket_num, 2)

        self.hash_wt = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(input_dim4lookup, bucket_num)
            ), requires_grad=True)  # embed weight. formula (6)
        if temperature <= 0:
            raise ValueError("temperature supposed to be in range (0,1])")
        self.register_buffer('temperature', torch.tensor(temperature))
        self.register_buffer('lambda_c', torch.tensor(lambda_c))
        self.register_buffer('epsilon', torch.tensor(epsilon))
        self.register_buffer('noise', torch.zeros(self.structure_logits.shape, dtype=torch.float))
        self.register_buffer('mask_choice', torch.tensor([[1.], [0.]]))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.bn = torch.nn.BatchNorm1d(input_dim4lookup * embed_dim, affine=False)

    def forward(self, embed_matrix, order, fix_structure):
        """Calculate grouped features' interaction score in specified `order`.

        :param embed_matrix: a batch of embedding features, tensor of size ``(batch_size, input_dim4lookup, embed_dim)``
        :type embed_matrix: torch.float
        :param order: order of feature interaction to be calculated.
        :type order: int
        :param fix_structure: whether to fix structure params during forward calculation
        :type fix_structure: bool
        :return: grouped features' interaction score
        :rtype: torch.float, size ``(batch, bucket_num*embed_dim)``(`order==1`),
            or size ``(batch, bucket_num)``(`order>1`)
        """
        if order < 1:
            raise ValueError("`order` should be a positive integer.")
        # bn for embed
        embed_matrix = self.bn(
            embed_matrix.view(
                -1, embed_matrix.shape[1] * embed_matrix.shape[2]
            )).view(*embed_matrix.shape)
        choices = self._differentiable_sampling(fix_structure)  # [input_dim4lookup, bucket_num]
        # high order fm formula
        comb_tmp = torch.matmul(
            torch.transpose(embed_matrix, 1, 2),
            torch.mul(choices, self.hash_wt)
        )  # [batch, k, bucket_num]
        comb = torch.pow(comb_tmp, order)
        # compensation if lambda_c != 0
        compensation = self.lambda_c * torch.matmul(
            torch.pow(
                torch.transpose(embed_matrix, 1, 2),
                order),  # [batch, k, input_dim4lookup]
            torch.pow(self.hash_wt, order)  # [input_dim4lookup, bucket_num]
        )  # [batch, k, bucket_num]
        comp_comb = comb - compensation
        if order == 1:
            return torch.reshape(comp_comb, (-1, comp_comb.shape[1] * comp_comb.shape[2]))
        reduced_comb = torch.sum(comp_comb, dim=1)
        return torch.reshape(reduced_comb, (-1, reduced_comb.shape[1]))

    def _differentiable_sampling(self, fix_structure):
        """Use Gumbel-Softmax trick to take argmax, while keeping differentiate w.r.t soft sample y.

        :param fix_structure: whether to fix structure params during forward calculation
        :type fix_structure: bool
        :return:
        """
        if fix_structure:
            logits = self.structure_logits
        else:
            noise = self.noise.uniform_(0.0, 1.0)
            logits = self.structure_logits - torch.log(-torch.log(noise + self.epsilon) + self.epsilon)
        y = self.softmax(logits / self.temperature)
        y_hard = torch.eq(y, torch.max(y, -1, keepdim=True).values).type(y.dtype)
        output = torch.matmul(
            (y_hard - y).detach() + y,
            self.mask_choice)  # [input_dim4lookup, bucket_num, 1]
        return output.squeeze(-1)
