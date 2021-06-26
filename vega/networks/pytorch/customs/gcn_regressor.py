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
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from vega.common import ClassType, ClassFactory


logger = logging.getLogger(__name__)


class GraphConvolution(nn.Module):
    """Graph Convolution Layer."""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of layer."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_, adj):
        """Forward function of graph convolution layer."""
        support = torch.matmul(input_, self.weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


@ClassFactory.register(ClassType.NETWORK)
class GCNRegressor(nn.Module):
    """Graph Convolution Network for regression."""

    def __init__(self, nfeat, ifsigmoid, layer_size=64):
        super(GCNRegressor, self).__init__()
        self.ifsigmoid = ifsigmoid
        self.size = layer_size
        self.gc1 = GraphConvolution(nfeat, self.size)
        self.gc2 = GraphConvolution(self.size, self.size)
        self.gc3 = GraphConvolution(self.size, self.size)
        self.gc4 = GraphConvolution(self.size, self.size)
        self.bn1 = nn.BatchNorm1d(self.size)
        self.bn2 = nn.BatchNorm1d(self.size)
        self.bn3 = nn.BatchNorm1d(self.size)
        self.bn4 = nn.BatchNorm1d(self.size)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.size, 1)
        self.init_weights()

    def init_weights(self):
        """Init parameters of each graph convolution layer in GCN."""
        nn.init.uniform_(self.gc1.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.gc2.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.gc3.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.gc4.weight, a=-0.05, b=0.05)

    def forward(self, input):
        """Forward function of GCN."""
        node_size = input.size()[1]
        adj, feat = input[:, :, :node_size], input[:, :, node_size:]
        x = F.relu(self.bn1(self.gc1(feat, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn2(self.gc2(x, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn3(self.gc3(x, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        x = F.relu(self.bn4(self.gc4(x, adj).transpose(2, 1)))
        x = x.transpose(1, 2)
        embeddings = x[:, x.size()[1] - 1, :]
        x = self.fc(embeddings)
        # if extract_embedding:
        #     return embeddings
        if self.ifsigmoid:
            return self.sigmoid(x)
        else:
            return x
