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
"""AutoGroup model file."""
import logging
import copy
import torch

from vega.common import ClassType, ClassFactory
from .fis.layers import LinearLayer, EmbeddingLayer, MultiLayerPerceptron, FeatureGroupLayer


class AttrProxy(object):
    """Translates index lookups into attribute lookups.

    :param module: an instance of torch module
    :type module: torch.nn.Module
    :param prefix: prefix of module name
    :type prefix: str
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        """Get items of model attribute.

        :param i: index of params
        :return: attribute of given index.
        """
        return getattr(self.module, self.prefix + str(i))


@ClassFactory.register(ClassType.NETWORK)
class AutoGroupModel(torch.nn.Module):
    """Automatic Feature Grouping.

    :param input_dim: feature space of dataset
    :type input_dim: int
    :param input_dim4lookup: feature number in `feature_id` and `feature_val`, usually equals to number of
    non-zero features in a single sample
    :type input_dim4lookup: int
    :param embed_dims: a list of `embed_dim`, each denoting length of feature embedding vector for
    corresponding `order`?`len(embed_dims) == max_order`
    :type embed_dims: list of int
    :param bucket_nums: a list of `bucket_num`, each denoting number of hash buckets for corresponding `order`.
     `len(bucket_nums) == max_order`
    :type bucket_nums: list of int
    :param temperature: temperature in Gumbel-Softmax
    :type temperature: float (0,1]
    :param lambda_c: compensation coefficient for feature interaction score
    :type lambda_c: float [0,1]
    :param max_order: maximum order of feature interaction
    :type max_order: int
    :param hidden_dims: width of each hidden layer, from bottom to top
    :type hidden_dims: list of int
    :param dropout_prob: dropout probability of all hidden layer
    :type dropout_prob: float
    :param batch_norm: applies batch normalization before activation, defaults to False
    :type batch_norm: bool, optional
    :param layer_norm: applies layer normalization before activation, defaults to False
    :type layer_norm: bool, optional
    """

    def __init__(self, **kwargs):
        """
        Construct the AutoGroupModel class.

        :param net_desc: config of the structure
        """
        super().__init__()
        self.desc = copy.deepcopy(kwargs)
        embed_dims = self.desc['embed_dims']
        bucket_nums = self.desc['bucket_nums']
        max_order = self.desc['max_order']
        if not len(embed_dims) == len(bucket_nums) == max_order:
            raise ValueError('Failed to automatic Feature Grouping.')
        self.linear = LinearLayer(self.desc['input_dim'])
        self.max_order = max_order
        for i in range(max_order):
            self.add_module('embedding_{}'.format(i), EmbeddingLayer(self.desc['input_dim'], embed_dims[i]))
            self.add_module('group_{}'.format(i),
                            FeatureGroupLayer(self.desc['input_dim4lookup'], embed_dims[i], bucket_nums[i],
                                              self.desc['temperature'], self.desc['lambda_c']))
        self.embedding = AttrProxy(self, 'embedding_')
        self.group = AttrProxy(self, 'group_')
        self.mlp_input_dim = embed_dims[0] * bucket_nums[0] + sum(bucket_nums[1:])
        self.mlp = MultiLayerPerceptron(
            self.mlp_input_dim, self.desc['hidden_dims'], self.desc['dropout_prob'],
            batch_norm=self.desc['batch_norm'], layer_norm=self.desc['layer_norm'])
        self.structure_params = [self.group[i].structure_logits for i in range(max_order)]
        self.net_params = []
        for i in range(max_order):
            self.net_params.append(self.group[i].hash_wt)
            self.net_params.extend(self.embedding[i].parameters())
        self.net_params.extend(self.linear.parameters())
        self.net_params.extend(self.mlp.parameters())
        logging.info("structure_params {} + net_params {} vs all params {}".format(
            len(self.structure_params), len(self.net_params), len(list(self.parameters()))
        ))

    def forward(self, feature_id, fix_structure=True):
        """Calculate logits of pctr for given batch of samples.

        :param feature_id: a batch of feature id, tensor of size ``(batch_size, input_dim4lookup)``
        :type feature_id: torch.int
        :param feature_val: a batch of feature value, tensor of size ``(batch_size, input_dim4lookup)``,
        defaults to None, i.e., all values equal to '1'
        :type feature_val: torch.float, optional
        :param fix_structure: whether to fix structure params during forward calculation
        :type fix_structure: bool
        :return: logits of pctr for given batch of samples
        :rtype: tensor.float, size ``(batch_size, 1)``
        """
        feature_val = None
        linear_score = self.linear(feature_id, feature_val).squeeze(1)
        # get each embedding output: embed -> bn -> group -> out
        mlp_in = list()
        for i, embed_layer, group_layer in zip(range(self.max_order), self.embedding, self.group):
            embed_v = embed_layer(feature_id, feature_val)
            group_feat = group_layer(embed_v, i + 1, fix_structure)
            mlp_in.append(group_feat)

        mlp_score = self.mlp(
            torch.cat(mlp_in, dim=1)
        ).squeeze(1)
        return linear_score + mlp_score
