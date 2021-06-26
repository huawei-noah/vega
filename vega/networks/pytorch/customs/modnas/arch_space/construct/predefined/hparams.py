# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Hyperparameter constructor."""
from modnas.registry.construct import register
from modnas.core.params import Numeric, Categorical


@register
class DefaultHParamSpaceConstructor():
    """Constructor that generates parameters from config."""

    def __init__(self, params):
        if isinstance(params, dict):
            params = params.items()
        elif isinstance(params, list):
            params = [(None, p) for p in params]
        self.params = params

    def __call__(self, model):
        """Run constructor."""
        del model
        for k, v in self.params:
            if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
                _ = Numeric(low=v[0][0], high=v[0][1], name=k)
            elif isinstance(v, list):
                _ = Categorical(choices=v, name=k)
            else:
                raise ValueError('support categorical and numeric hparams only')
