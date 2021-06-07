# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Bayesian Optimizer based on scikit-optimize."""
import time
import numpy as np
from collections import OrderedDict
from modnas.registry.optim import register
from modnas.optim.base import OptimBase
from modnas.core.params import Categorical as ParamCategorical, Numeric
try:
    import skopt
    from skopt import Optimizer
    from skopt.space import Real, Integer, Categorical
except ImportError:
    skopt = None


@register
class SkoptOptim(OptimBase):
    """Scikit-optimize Optimizer class."""

    def __init__(self, skopt_args=None, space=None):
        super().__init__(space)
        if skopt is None:
            raise ValueError('scikit-optimize is not installed')
        skopt_dims = []
        param_names = []
        for n, p in self.space.named_params():
            if isinstance(p, Numeric):
                if p.is_int():
                    sd = Integer(*p.bound, name=n)
                else:
                    sd = Real(*p.bound, name=n)
            elif isinstance(p, ParamCategorical):
                sd = Categorical(p.choices, name=n)
            else:
                continue
            skopt_dims.append(sd)
            param_names.append(n)
        skopt_args = skopt_args or {}
        skopt_args['dimensions'] = skopt_dims
        if 'random_state' not in skopt_args:
            skopt_args['random_state'] = int(time.time())
        self.param_names = param_names
        self.skoptim = Optimizer(**skopt_args)

    def has_next(self):
        """Return True if Optimizer has the next set of parameters."""
        return True

    def convert_param(self, p):
        """Return value converted from scikit-optimize space."""
        if isinstance(p, np.float):
            return float(p)
        return p

    def _next(self):
        """Return the next set of parameters."""
        next_pt = self.skoptim.ask()
        next_params = OrderedDict()
        for n, p in zip(self.param_names, next_pt):
            next_params[n] = self.convert_param(p)
        return next_params

    def next(self, batch_size):
        """Return the next batch of parameter sets."""
        if batch_size == 1:
            return [self._next()]
        next_pts = self.skoptim.ask(n_points=batch_size)
        next_params = []
        for pt in next_pts:
            params = OrderedDict()
            for n, p in zip(self.param_names, pt):
                params[n] = self.convert_param(p)
            next_params.append(params)
        return next_params

    def step(self, estim):
        """Update Optimizer states using Estimator evaluation results."""
        def to_metrics(res):
            if isinstance(res, dict):
                return list(res.values())[0]
            if isinstance(res, (tuple, list)):
                return res[0]
            return res

        inputs, results = estim.get_last_results()
        skinputs = [list(inp.values()) for inp in inputs]
        skresults = [-to_metrics(r) for r in results]
        self.skoptim.tell(skinputs, skresults)
