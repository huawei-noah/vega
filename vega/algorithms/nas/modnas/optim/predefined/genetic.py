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

"""Genetic search algorithms."""
import random
from collections import OrderedDict
from typing import Callable, Dict, List, Union, Optional
import numpy as np
from modnas.registry.optim import register
from modnas.core.param_space import ParamSpace
from modnas.estim.base import EstimBase
from ..base import CategoricalSpaceOptim


class GeneticOptim(CategoricalSpaceOptim):
    """Optimizer with genetic operators on a population."""

    def __init__(self, pop_size: int, max_it: int = 1000, space: Optional[ParamSpace] = None) -> None:
        super().__init__(space)
        self.max_it = max_it
        self.pop_size = pop_size
        self.operators = []
        self.metrics = []
        self.population = self._initialize()

    def _initialize(self):
        raise NotImplementedError

    def _mating(self, pop: List[OrderedDict]) -> List[OrderedDict]:
        cur_pop = pop
        for op in self.operators:
            cur_pop = op(cur_pop)
        return cur_pop

    def _next(self) -> OrderedDict:
        params = self.population[len(self.metrics)]
        self.set_visited_params(params)
        return params

    def add_operator(self, operator: Callable) -> None:
        """Add a genetic operator."""
        self.operators.append(operator)

    def to_metrics(self, res: Union[float, Dict[str, float]]) -> float:
        """Return scalar metrics from evaluation results."""
        if isinstance(res, dict):
            return list(res.values())[0]
        if isinstance(res, (tuple, list)):
            return res[0]
        return res

    def step(self, estim: EstimBase) -> None:
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
                 pop_size: int = 100,
                 n_parents: int = 2,
                 n_offsprings: int = 1,
                 n_select: int = 10,
                 n_eliminate: int = 1,
                 n_crossover: Optional[int] = None,
                 mutation_prob: float = 0.01,
                 space: Optional[ParamSpace] = None) -> None:
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

    def _initialize(self) -> List[OrderedDict]:
        return [self.get_random_params() for _ in range(self.pop_size)]

    def _survival(self, pop):
        n_survival = len(pop) - self.n_eliminate
        if n_survival >= len(pop):
            return pop
        metrics = np.array(self.metrics)
        idx = np.argpartition(metrics, -n_survival)[-n_survival:]
        self.metrics = [metrics[i] for i in idx]
        return [pop[i] for i in idx]

    def _selection(self, pop: List[OrderedDict]) -> List[OrderedDict]:
        n_select = self.n_select
        if n_select >= len(pop):
            return pop
        metrics = np.array(self.metrics)
        idx = np.argpartition(metrics, -n_select)[-n_select:]
        self.metrics = [metrics[i] for i in idx]
        return [pop[i] for i in idx]

    def _crossover(self, pop: List[OrderedDict]) -> List[OrderedDict]:
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

    def _mutation(self, pop: List[OrderedDict]) -> List[OrderedDict]:
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

    def _survival(self, pop: List[OrderedDict]) -> List[OrderedDict]:
        s_idx = self.n_eliminate
        if s_idx <= 0:
            return pop
        self.metrics = self.metrics[s_idx:]
        return pop[s_idx:]
