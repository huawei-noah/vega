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

"""CARS and DARTS network."""

from vega.common import ClassFactory, ClassType
from vega.modules.blocks import AuxiliaryHead
from vega.modules.connections import Cells
from vega.modules.module import Module
from mindspore import ops
import numpy as np
from mindspore import Tensor


@ClassFactory.register(ClassType.NETWORK)
class DartsNetwork(Module):
    """Create Darts SearchSpace."""

    def __init__(self, stem, cells, head, init_channels, num_classes, auxiliary, search, aux_size=8,
                 auxiliary_layer=13, drop_path_prob=0):
        """Create layers."""
        super(DartsNetwork, self).__init__()
        self.is_search = search
        self._auxiliary = auxiliary
        self.drop_path_prob = drop_path_prob
        if auxiliary:
            self._aux_size = aux_size
            self._auxiliary_layer = auxiliary_layer
        # Build stems part
        self.pre_stems = ClassFactory.get_instance(ClassType.NETWORK, stem)
        # Build cells part
        c_curr = self.pre_stems.output_channel
        self.cells_ = Cells(cells, c_curr, init_channels, auxiliary=auxiliary, auxiliary_layer=auxiliary_layer)
        # output params
        self.len_alpha = self.cells_.len_alpha
        self.num_ops = self.cells_.num_ops
        self.steps = self.cells_.steps
        c_prev, c_aux = self.cells_.output_channels()
        if not search and auxiliary:
            self.auxiliary_head = AuxiliaryHead(c_aux, num_classes, aux_size)
        # head
        self.head = ClassFactory.get_instance(
            ClassType.NETWORK, head, base_channel=c_prev, num_classes=num_classes)

        # Initialize architecture parameters
        self.set_parameters(
            'alphas_normal',
            1e-3 * Tensor(np.random.randn(self.len_alpha, self.num_ops).astype(np.float32)))
        self.set_parameters(
            'alphas_reduce',
            1e-3 * Tensor(np.random.randn(self.len_alpha, self.num_ops).astype(np.float32)))

        self.cell_list = self.cells_.children()
        self.name_list = []
        for tmp_cell in self.cells_.children():
            self.name_list.append(tmp_cell.__class__.__name__)

    @property
    def learnable_params(self):
        """Get learnable params of alphas."""
        return [self.alphas_normal, self.alphas_reduce]

    @property
    def arch_weights(self):
        """Get weights of alphas."""
        self.alphas_normal = self.get_weights('alphas_normal')
        self.alphas_reduce = self.get_weights('alphas_reduce')
        softmax = ops.Softmax()
        alphas_normal = softmax(self.alphas_normal)
        softmax = ops.Softmax()
        alphas_reduce = softmax(self.alphas_reduce)
        return [alphas_normal.asnumpy(), alphas_reduce.asnumpy()]

    def get_weight_ops(self):
        """Get weight ops."""
        return super().get_weight_ops('alphas')

    def calc_alphas(self, alphas, dim=-1, **kwargs):
        """Calculate Alphas."""
        softmax = ops.Softmax()
        return softmax(alphas)

    def call(self, input, alpha=None):
        """Forward a model that specified by alpha.

        :param input: An input tensor
        :type input: Tensor
        """
        s0, s1 = self.pre_stems(input)
        alphas_normal, alphas_reduce = self.alphas_normal, self.alphas_reduce
        if alpha is not None:
            alphas_normal, alphas_reduce = alpha[:self.len_alpha], alpha[self.len_alpha:]
        else:
            alphas_normal = self.calc_alphas(alphas_normal)
            alphas_reduce = self.calc_alphas(alphas_reduce)
        logits_aux = None
        for i, cell in enumerate(self.cell_list):
            weights = None
            weights = alphas_reduce if self.name_list[i] == 'NormalCell' and self.is_search else weights
            weights = alphas_normal if self.name_list[i] == 'ReduceCell' and self.is_search else weights
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
            if not self.is_search and self._auxiliary and i == self._auxiliary_layer:
                logits_aux = self.auxiliary_head(s1)
        logits = self.head(s1)
        if logits_aux is not None:
            return logits, logits_aux
        else:
            return logits


@ClassFactory.register(ClassType.NETWORK)
class CARSDartsNetwork(DartsNetwork):
    """Base CARS-Darts Network of classification."""

    def __init__(self, stem, cells, head, init_channels, num_classes=10, auxiliary=False, search=True, aux_size=8,
                 auxiliary_layer=13, drop_path_prob=0.):
        """Init CARSDartsNetwork."""
        super(CARSDartsNetwork, self).__init__(stem, cells, head, init_channels, num_classes, auxiliary, search,
                                               aux_size, auxiliary_layer, drop_path_prob)


@ClassFactory.register(ClassType.NETWORK)
class GDASDartsNetwork(DartsNetwork):
    """Base GDAS-DARTS Network of classification."""

    def __init__(self, stem, cells, head, init_channels, num_classes=10, auxiliary=False, search=True, aux_size=8,
                 auxiliary_layer=13, drop_path_prob=0.):
        """Init GDASDartsNetwork."""
        super(GDASDartsNetwork, self).__init__(stem, cells, head, init_channels, num_classes, auxiliary, search,
                                               aux_size, auxiliary_layer, drop_path_prob)

    def calc_alphas(self, alphas, dim=-1, tau=1, hard=True, eps=1e-10):
        """Calculate Alphas."""
        return ops.gumbel_softmax(alphas, tau, hard, eps, dim)
