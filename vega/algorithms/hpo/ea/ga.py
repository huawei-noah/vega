# -*- coding: utf-8 -*-

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

"""Evolution."""

import random
import logging
import numpy as np
from vega.common.pareto_front import get_pareto_index


logger = logging.getLogger(__name__)


class GeneticAlgorithm(object):
    """Evolution."""

    def __init__(self, search_space=None, random_samples=32, prob_crossover=0.6, prob_mutatation=0.2):
        """Init for Evolution."""
        self.search_space = search_space
        self.random_samples = random_samples
        self.prob_crossover = prob_crossover
        self.prob_mutatation = prob_mutatation
        self.scores = {}
        self.sample_count = 0

    def add(self, config, score):
        """Add feature and label to train model.

        :param feature:
        :param label:
        :return:
        """
        found = False
        for value in self.scores.values():
            if value["config"] == config:
                value["score"] = score
                found = True
                break
        if not found:
            self.sample_count += 1
            self.scores[self.sample_count] = {"config": config, "score": score}

    def propose(self, num=1):
        """Propose hyper-parameters to json.

        :param num: int, number of hps to propose, default is 1.
        :return: list
        """
        if self.sample_count < max(self.random_samples, 2):
            self.sample_count += 1
            config = self.search_space.sample()
            self.scores[self.sample_count] = {"config": config}
            logger.info(f"propose random sample: {config}")
            return [config]

        config = self.evolute()
        if config:
            self.sample_count += 1
            self.scores[self.sample_count] = {"config": config}
            logger.info(f"propose evaluated sample: {config}")
            return [config]
        else:
            return None

    def evolute(self):
        """Evolution."""
        inds = self.selection()
        if not inds:
            return None
        ind = self.crossover(inds[0], inds[1], self.prob_crossover)
        ind = self.mutatation(ind, self.prob_mutatation)
        ind = self.search_space.verify_constraints(ind)
        return ind

    def selection(self):
        """Select pareto front individual."""
        ids = [id for id, value in self.scores.items() if "score" in value]
        if len(ids) == 0:
            return None
        rewards = [self.scores[id]["score"] for id in ids]
        indexes = get_pareto_index(np.array(rewards)).tolist()
        pareto = [id for i, id in enumerate(ids) if indexes[i]]

        if len(pareto) < 2:
            others = [id for id in self.scores if id not in pareto]
            pareto = pareto + random.sample(others, 2 - len(pareto))
        else:
            pareto = random.sample(pareto, 2)

        return [value["config"] for id, value in self.scores.items() if id in pareto]

    def crossover(self, ind0, ind1, prob=0.6):
        """Cross over operation in EA algorithm.

        :param ind0: individual 0
        :type ind0: congfig dict
        :param ind1: individual 1
        :type ind1: config dict
        :return: new individual
        :rtype: config dict
        """
        ind = ind0.copy()
        for item in ind:
            if item in ind1 and random.random() > 1 - prob:
                ind[item] = ind1[item]
        for item in ind1:
            if item not in ind and random.random() > 1 - prob:
                ind[item] = ind1[item]
        return ind

    def mutatation(self, ind0, prob=0.2):
        """Mutate operation in EA algorithm.

        :param ind: individual
        :type ind: config dict
        :return: new individual
        :rtype: config dict
        """
        ind1 = self.search_space.sample()
        ind = self.crossover(ind0, ind1, prob=prob)
        return ind
