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

"""SEARCH_ALGORITHM for searching quantization model."""
import logging
import random
import numpy as np
from vega.common import ClassFactory, ClassType
from vega.report import ReportServer
from vega.core.search_algs import SearchAlgorithm
from .conf import QuantConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class QuantEA(SearchAlgorithm):
    """Class of Evolution Algorithm used to Quant Example.

    :param search_space: search space
    :type search_space: SearchSpace
    """

    config = QuantConfig()

    def __init__(self, search_space, **kwargs):
        super(QuantEA, self).__init__(search_space, **kwargs)
        self.length = self.config.policy.length
        self.num_individual = self.config.policy.num_individual
        self.num_generation = self.config.policy.num_generation
        self.random_samples = self.config.policy.random_samples
        self.bit_candidates = self.search_space.bit_candidates
        self.random_count = 0
        self.ea_count = 0
        self.ea_epoch = 0

    def crossover(self, ind0, ind1):
        """Cross over operation in EA algorithm.

        :param ind0: individual 0
        :type ind0: list of int
        :param ind1: individual 1
        :type ind1: list of int
        :return: new individual 0, new individual 1
        :rtype: list of int, list of int
        """
        two_idxs = np.random.randint(0, self.length, 2)
        start_idx, end_idx = np.min(two_idxs), np.max(two_idxs)
        a_copy = ind0.copy()
        b_copy = ind1.copy()
        a_copy[start_idx: end_idx] = ind1[start_idx: end_idx]
        b_copy[start_idx: end_idx] = ind0[start_idx: end_idx]
        return a_copy, b_copy

    def mutatation(self, ind):
        """Mutate operation in EA algorithm.

        :param ind: individual
        :type ind: list of int
        :return: new individual
        :rtype: list of int
        """
        two_idxs = np.random.randint(0, self.length, 2)
        start_idx, end_idx = np.min(two_idxs), np.max(two_idxs)
        a_copy = ind.copy()
        for k in range(start_idx, end_idx):
            candidates_ = self.bit_candidates.copy()
            candidates_.remove(ind[k])
            a_copy[k] = random.choice(candidates_)
        return a_copy

    def search(self):
        """Search one NetworkDesc from search space.

        :return: search id, network desc
        :rtype: int, NetworkDesc
        """
        if self.random_count < self.random_samples:
            desc = self._random_sample()
            return self.random_count, desc
        records = ReportServer().get_pareto_front_records(self.step_name, self.num_individual)
        codes = [record.desc.get('nbit_w_list') + record.desc.get('nbit_a_list') for record in records]
        logging.info("codes=%s", codes)
        if len(codes) < 2:
            encoding1, encoding2 = codes[0], codes[0]
        else:
            encoding1, encoding2 = random.sample(codes, 2)
        choice = random.randint(0, 1)
        # mutate
        if choice == 0:
            encoding_new = self.mutatation(encoding1)
        # crossover
        else:
            encoding_new, _ = self.crossover(encoding1, encoding2)
        self.ea_count += 1
        if self.ea_count % self.num_individual == 0:
            self.ea_epoch += 1
        desc = self.codec.decode(encoding_new)
        return self.random_count + self.ea_count, desc

    def _random_sample(self):
        """Choose one sample randomly.

        :return: network desc
        :rtype: NetworkDesc
        """
        individual = []
        for _ in range(self.length):
            individual.append(random.choice(self.bit_candidates))
        self.random_count += 1
        base_desc = self._random_base_desc()
        desc = self.codec.decode(individual)
        desc.update(base_desc)
        return desc

    def _random_base_desc(self):
        if not self.search_space_list:
            return dict()
        else:
            return random.choice(self.search_space_list)

    @property
    def is_completed(self):
        """Whether to complete algorithm.

        :return: whether to complete algorithm
        :rtype: bool
        """
        return self.random_count >= self.random_samples and self.ea_epoch >= self.num_generation

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.num_individual * self.num_generation + self.random_samples
