# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Genetic search algorithms."""
import numpy as np
import random
from ..base import CategoricalSpaceOptim
from modnas.registry.optim import register


class GeneticOptim(CategoricalSpaceOptim):
    """Optimizer with genetic operators on a population."""

    def __init__(self, pop_size, max_it=1000, space=None):
        super().__init__(space)
        self.max_it = max_it
        self.pop_size = pop_size
        self.operators = []
        self.metrics = []
        self.population = self._initialize()

    def _initialize(self):
        raise NotImplementedError

    def _mating(self, pop):
        cur_pop = pop
        for op in self.operators:
            cur_pop = op(cur_pop)
        return cur_pop

    def _next(self):
        params = self.population[len(self.metrics)]
        self.set_visited_params(params)
        return params

    def add_operator(self, operator):
        """Add a genetic operator."""
        self.operators.append(operator)

    def to_metrics(self, res):
        """Return scalar metrics from evaluation results."""
        if isinstance(res, dict):
            return list(res.values())[0]
        if isinstance(res, (tuple, list)):
            return res[0]
        return res

    def step(self, estim):
        """Update Optimizer states using Estimator evaluation results."""
        _, results = estim.get_last_results()
        results = [self.to_metrics(res) for res in results]
        self.metrics.extend(results)
        if len(self.metrics) >= len(self.population):
            self.population = self._mating(self.population)
            self.metrics = []


@register
class EvolutionOptim(GeneticOptim):
    """Optimizer with Evolution algorithm."""

    def __init__(self,
                 pop_size=100,
                 n_parents=2,
                 n_offsprings=1,
                 n_select=10,
                 n_eliminate=1,
                 n_crossover=None,
                 mutation_prob=0.01,
                 space=None):
        super().__init__(space=space, pop_size=pop_size)
        self.add_operator(self._survival)
        self.add_operator(self._selection)
        self.add_operator(self._crossover)
        self.add_operator(self._mutation)
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings
        self.n_select = pop_size if n_select is None else n_select
        self.n_eliminate = 0 if n_eliminate is None else n_eliminate
        self.n_crossover = pop_size if n_crossover is None else n_crossover
        self.mutation_prob = mutation_prob

    def _initialize(self):
        return [self.get_random_params() for _ in range(self.pop_size)]

    def _survival(self, pop):
        n_survival = len(pop) - self.n_eliminate
        if n_survival >= len(pop):
            return pop
        metrics = np.array(self.metrics)
        idx = np.argpartition(metrics, -n_survival)[-n_survival:]
        self.metrics = [metrics[i] for i in idx]
        return [pop[i] for i in idx]

    def _selection(self, pop):
        n_select = self.n_select
        if n_select >= len(pop):
            return pop
        metrics = np.array(self.metrics)
        idx = np.argpartition(metrics, -n_select)[-n_select:]
        self.metrics = [metrics[i] for i in idx]
        return [pop[i] for i in idx]

    def _crossover(self, pop):
        next_pop = []
        it = 0
        while len(next_pop) < self.n_crossover and it < self.max_it:
            parents = [random.choice(pop) for _ in range(self.n_parents)]
            for _ in range(self.n_offsprings):
                n_gene = parents[0].copy()
                for name in parents[0]:
                    values = [p[name] for p in parents]
                    n_gene[name] = random.choice(values)
                if self.is_visited_params(n_gene):
                    continue
                next_pop.append(n_gene)
            it += 1
        while len(next_pop) < self.n_crossover:
            next_pop.append(self.get_random_params())
        return next_pop

    def _mutation(self, pop):
        next_pop = []
        for gene in pop:
            it = 0
            while it < self.max_it:
                m_gene = gene.copy()
                for name, value in gene.items():
                    p = self.space.get_param(name)
                    if random.random() < self.mutation_prob:
                        nidx = idx = p.get_index(value)
                        while nidx == idx:
                            nidx = random.randint(0, len(p) - 1)
                        m_gene[name] = p.get_value(nidx)
                if not self.is_visited_params(m_gene):
                    break
            if it == self.max_it:
                m_gene = self.get_random_params()
            next_pop.append(m_gene)
        return next_pop


@register
class RegularizedEvolutionOptim(EvolutionOptim):
    """Optimizer with Regularized Evolution algorithm."""

    def _survival(self, pop):
        s_idx = self.n_eliminate
        if s_idx <= 0:
            return pop
        self.metrics = self.metrics[s_idx:]
        return pop[s_idx:]
