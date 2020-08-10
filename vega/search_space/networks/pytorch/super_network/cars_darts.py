# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""SuperNet for CARS-DARTS."""
import logging
import torch
from vega.search_space.networks import NetworkFactory, NetTypes
from vega.search_space.networks.pytorch.super_network import DartsNetwork
import numpy as np

logger = logging.getLogger(__name__)


@NetworkFactory.register(NetTypes.SUPER_NETWORK)
class CARSDartsNetwork(DartsNetwork):
    """Base CARS-Darts Network of classification.

    :param desc: darts description
    :type desc: Config
    """

    def __init__(self, desc):
        """Init CARSDartsNetwork."""
        super(CARSDartsNetwork, self).__init__(desc)
        self.num_ops = self.num_ops()
        self.steps = self.desc.normal.steps
        self.len_alpha = self.len_alpha()

    def len_alpha(self):
        """Get length of alpha."""
        k_normal = len(self.desc.normal.genotype)
        return k_normal

    def num_ops(self):
        """Get number of candidate operations."""
        num_ops = len(self.desc.normal.genotype[0][0])
        return num_ops

    def forward(self, input, alpha):
        """Forward a model that specified by alpha.

        :param input: An input tensor
        :type input: Tensor
        """
        alphas_normal = alpha[:self.len_alpha]
        alphas_reduce = alpha[self.len_alpha:]
        s0, s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if self.search:
                if self.desc.network[i + 1] == 'reduce':
                    weights = alphas_reduce
                else:
                    weights = alphas_normal
            else:
                weights = None
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
            if not self.search:
                if self._auxiliary and i == self._auxiliary_layer:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self._auxiliary and not self.search:
            return logits, logits_aux
        else:
            return logits
