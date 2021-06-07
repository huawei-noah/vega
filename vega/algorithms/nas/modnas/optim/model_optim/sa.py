# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Simulated annealing model optimum finder."""
import heapq
import random
import numpy as np
from .base import ModelOptim
from modnas.registry.model_optim import register
from modnas.utils.logging import get_logger


logger = get_logger('model_optim')


@register
class SimulatedAnnealingModelOptim(ModelOptim):
    """Simulated annealing model optimum finder class."""

    def __init__(self,
                 space,
                 temp_init=1e4,
                 temp_end=1e-4,
                 cool=0.95,
                 cool_type='exp',
                 batch_size=128,
                 n_iter=1,
                 keep_history=True):
        super().__init__(space)
        self.temp_init = temp_init
        self.temp_end = temp_end
        self.cool = cool
        self.cool_type = cool_type
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.keep_history = keep_history
        self.history = None

    def disturb(self, params):
        """Return randomly disturbed parameter."""
        pname = list(params)[random.randint(0, len(params) - 1)]
        p = self.space.get_param(pname)
        nidx = idx = p.get_index(params[pname])
        while nidx == idx:
            nidx = random.randint(0, len(p) - 1)
        new_params = params.copy()
        new_params[pname] = p.get_value(nidx)
        return new_params

    def get_optimums(self, model, size, excludes):
        """Return optimums in score model."""
        topq = []
        for _ in range(self.n_iter):
            self.run_sa(model, size, excludes, topq)
        return [item[-1] for item in topq[::1]]

    def run_sa(self, model, size, excludes, topq):
        """Run SA algorithm."""
        if self.history is None:
            params = [self.get_random_params(excludes) for _ in range(self.batch_size)]
        else:
            params = self.history
        results = model.predict(params)

        for r, p in zip(results, params):
            pi = self.space.get_categorical_index(p)
            if len(topq) < size:
                heapq.heappush(topq, (r, pi))
            elif r > topq[0][0]:
                heapq.heapreplace(topq, (r, pi))

        temp = self.temp_init
        temp_end = self.temp_end
        cool = self.cool
        cool_type = self.cool_type
        while (temp > temp_end):
            next_params = [self.disturb(p) for p in params]
            next_results = model.predict(next_params)

            for r, p in zip(next_results, next_params):
                pi = self.space.get_categorical_index(p)
                if pi in excludes:
                    continue
                if len(topq) < size:
                    heapq.heappush(topq, (r, pi))
                elif r > topq[0][0]:
                    heapq.heapreplace(topq, (r, pi))

            ac_prob = np.minimum(np.exp((next_results - results) / (temp + 1e-7)), 1.)
            for i in range(self.batch_size):
                if random.random() < ac_prob[i]:
                    params[i] = next_params[i]
                    results[i] = next_results[i]

            if cool_type == 'exp':
                temp *= cool
            elif cool_type == 'linear':
                temp -= cool
            logger.debug('SA: temp: {:.4f} max: {:.4f}'.format(temp, np.max(next_results)))

        if self.keep_history:
            self.history = params
