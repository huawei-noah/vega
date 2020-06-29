# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Darts operators."""
import numpy as np
import torch
import torch.nn as nn
from vega.core.common.class_factory import ClassType, ClassFactory
import torch.nn.functional as F


@ClassFactory.register(ClassType.SEARCH_SPACE)
class DartsAlphas(object):
    """Initialize alphas."""

    def __init__(self, genotype_size, num_ops):
        super(DartsAlphas, self).__init__()
        self.k = genotype_size
        self.num_ops = num_ops

    def _initialize_alphas(self, model):
        """Initialize architecture parameters."""
        model.register_buffer('alphas_normal',
                              (1e-3 * torch.randn(self.k, self.num_ops)).cuda().requires_grad_())
        model.register_buffer('alphas_reduce',
                              (1e-3 * torch.randn(self.k, self.num_ops)).cuda().requires_grad_())
        self._arch_parameters = [
            model.alphas_normal,
            model.alphas_reduce,
        ]
        return model

    def __call__(self, model):
        """Generate new model."""
        dist_model = self._initialize_alphas(model)
        weights_normal = F.softmax(
            dist_model.alphas_normal, dim=-1).data.cpu().numpy()
        weights_reduce = F.softmax(
            dist_model.alphas_reduce, dim=-1).data.cpu().numpy()
        arch_weights = [weights_normal, weights_reduce]
        dist_model.arch_weights = arch_weights
        dist_model.arch_parameters = self._arch_parameters
        return dist_model
