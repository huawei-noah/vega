# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Initialization constructor."""
import numpy as np
from modnas.registry.construct import register
from modnas.core.param_space import ParamSpace


@register
class DefaultInitConstructor():
    """Constructor that initializes the architecture space."""

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, model):
        """Run constructor."""
        ParamSpace().reset()
        seed = self.seed
        if seed:
            np.random.seed(seed)
        return model
