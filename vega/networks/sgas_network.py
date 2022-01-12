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

"""SGAS network."""
from vega.common import ClassFactory, ClassType
from vega.modules.operators import ops
from vega.networks.super_network import DartsNetwork
import torch
from torch.autograd import Variable


@ClassFactory.register(ClassType.NETWORK)
class SGASNetwork(DartsNetwork):
    """Base GDAS-DARTS Network of classification."""

    def __init__(self, stem, cells, head, init_channels, num_classes=10, auxiliary=False, search=True, aux_size=8,
                 auxiliary_layer=13, drop_path_prob=0.):
        """Init SGASNetwork."""
        super(SGASNetwork, self).__init__(stem, cells, head, init_channels, num_classes, auxiliary, search,
                                          aux_size, auxiliary_layer, drop_path_prob)
        self.normal_selected_idxs = None
        self.reduce_selected_idxs = None
        self.normal_candidate_flags = None
        self.reduce_candidate_flags = None
        self.initialize()

    def initialize(self):
        """Initialize architecture parameters."""
        self.alphas_normal = []
        self.alphas_reduce = []
        for i in range(self.steps):
            for n in range(2 + i):
                self.alphas_normal.append(Variable(
                    ops.random_normal(self.num_ops).cuda() / self.num_ops, requires_grad=True))
                self.alphas_reduce.append(Variable(
                    ops.random_normal(self.num_ops).cuda() / self.num_ops, requires_grad=True))

    @property
    def learnable_params(self):
        """Get learnable params of alphas."""
        return self.alphas_normal + self.alphas_reduce

    @property
    def arch_weights(self):
        """Get weights of alphas."""
        self.alphas_normal = self.get_weights('alphas_normal')
        self.alphas_reduce = self.get_weights('alphas_reduce')
        alphas_normal = ops.softmax(torch.stack(self.alphas_normal, dim=0), -1)
        alphas_reduce = ops.softmax(torch.stack(self.alphas_reduce, dim=0), -1)
        return [ops.to_numpy(alphas_normal), ops.to_numpy(alphas_reduce)]

    def calc_alphas(self, alphas, dim=-1, **kwargs):
        """Calculate Alphas."""
        new_alphas = []
        for alpha in alphas:
            new_alphas.append(ops.softmax(alpha, dim))
        return new_alphas

    def call(self, input, alpha=None):
        """Forward a model that specified by alpha.

        :param input: An input tensor
        :type input: Tensor
        """
        # TODO: training for tf
        self.initialize()
        s0, s1 = self.pre_stems(input)
        alphas_normal, alphas_reduce = self.alphas_normal, self.alphas_reduce
        if alpha is not None:
            alphas_normal, alphas_reduce = alpha[:self.len_alpha], alpha[self.len_alpha:]
        else:
            alphas_normal = self.calc_alphas(alphas_normal)
            alphas_reduce = self.calc_alphas(alphas_reduce)
        logits_aux = None
        for i, cell in enumerate(self.cells_.children()):
            weights = None
            selected_idxs = None
            if self.is_search:
                if cell.__class__.__name__ == 'NormalCell':
                    weights = alphas_normal
                    selected_idxs = self.normal_selected_idxs
                elif cell.__class__.__name__ == 'ReduceCell':
                    weights = alphas_reduce
                    selected_idxs = self.reduce_selected_idxs
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob, selected_idxs)
            if not self.is_search and self._auxiliary and i == self._auxiliary_layer:
                logits_aux = self.auxiliary_head(s1)
        logits = self.head(s1)
        if logits_aux is not None:
            return logits, logits_aux
        else:
            return logits
