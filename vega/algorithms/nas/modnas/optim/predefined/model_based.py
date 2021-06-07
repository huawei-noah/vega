# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Model-based Optimizer."""
from ..base import CategoricalSpaceOptim
from modnas.registry.score_model import build as build_score_model
from modnas.registry.model_optim import build as build_model_optim
from modnas.registry.optim import register


@register
class ModelBasedOptim(CategoricalSpaceOptim):
    """Model-based Optimizer class."""

    def __init__(self, model_config, model_optim_config, greedy_e=0.05, n_next_pts=32, space=None):
        super().__init__(space)
        self.model = build_score_model(model_config, space=self.space)
        self.model_optim = build_model_optim(model_optim_config, space=self.space)
        self.n_next_pts = n_next_pts
        self.greedy_e = greedy_e
        self.train_x = []
        self.train_y = []
        self.next_xs = []
        self.next_pt = 0
        self.train_ct = 0

    def _next(self):
        while self.next_pt < len(self.next_xs):
            index = self.next_xs[self.next_pt]
            if not self.is_visited(index):
                break
            self.next_pt += 1
        if self.next_pt >= len(self.next_xs) - int(self.greedy_e * self.n_next_pts):
            index = self.get_random_index()
        self.set_visited(index)
        return self.space.get_categorical_params(index)

    def step(self, estim):
        """Update Optimizer states using Estimator evaluation results."""
        inputs, results = estim.get_last_results()
        for inp, res in zip(inputs, results):
            self.train_x.append(inp)
            self.train_y.append(res)
        if len(self.train_x) < self.n_next_pts * (self.train_ct + 1):
            return
        self.model.fit(self.train_x, self.train_y)
        self.next_xs = self.model_optim.get_optimums(self.model, self.n_next_pts, self.visited)
        self.next_pt = 0
        self.train_ct += 1
