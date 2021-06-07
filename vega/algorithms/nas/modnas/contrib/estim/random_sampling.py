# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Uniformly samples and trains subnets."""
import random
from modnas.estim.predefined.default import DefaultEstim
from modnas.registry.estim import register
from modnas.core.param_space import ParamSpace


@register
class RandomSamplingEstim(DefaultEstim):
    """Trains a subnet uniformly sampled from the supernet in each step."""

    def __init__(self, *args, seed=1, save_best=True, **kwargs):
        super().__init__(*args, save_best=save_best, **kwargs)
        random.seed(seed)
        self.space_size = ParamSpace().categorical_size()

    def loss(self, data, output=None, model=None, mode=None):
        """Sample a subnet and compute its loss."""
        loss = super().loss(data, output, model, mode)
        params = ParamSpace().get_categorical_params(random.randint(0, self.space_size - 1))
        ParamSpace().update_params(params)
        return loss
