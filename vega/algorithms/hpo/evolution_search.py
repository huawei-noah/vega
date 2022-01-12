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

"""EvolutionAlgorithm."""
from collections import OrderedDict
import random
import logging
import numpy as np
from vega.core.search_algs import SearchAlgorithm
from vega.common import ClassFactory, ClassType
from vega.report import ReportServer
from .evolution_conf import EvolutionConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class EvolutionAlgorithm(SearchAlgorithm):
    """EvolutionAlgorithm."""

    config = EvolutionConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init for EvolutionAlgorithm."""
        super(EvolutionAlgorithm, self).__init__(search_space, **kwargs)
        self._code_cache = OrderedDict()
        self.sample_count = 0
        self.num_individual = self.config.policy.num_individual
        self.num_generation = self.config.policy.num_generation
        self.random_samples = self.config.policy.random_samples

    def search(self):
        """Search one NetworkDesc from search space.

        :return: search id, network desc
        :rtype: int, NetworkDesc
        """
        if self.sample_count < self.random_samples:
            self.sample_count += 1
            desc = self.search_space.sample()
            sample = dict(worker_id=self.sample_count, encoded_desc=desc)
            self._code_cache[self.sample_count] = desc
            return sample
        records = ReportServer().get_pareto_front_records(self.step_name, self.num_individual)
        if not records:
            return None
        codes = []
        each_codes_cache = {}
        # Merge codes
        for record in records:
            each_code = []
            for key, item in self._code_cache.get(record.worker_id).items():
                if isinstance(item, int):
                    item = [item]
                each_codes_cache[key] = len(item)
                each_code.extend(item)
            codes.append(each_code)
        self.length = len(codes[0])
        logging.info("codes sum={}, code length={}".format(sum(codes[0]), self.length))
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
        # split codes
        desc = {}
        for _name, _size in each_codes_cache.items():
            desc[_name] = encoding_new[:_size]
            encoding_new = encoding_new[_size:]
        self.sample_count += 1
        sample = dict(worker_id=self.sample_count, encoded_desc=desc)
        self._code_cache[self.sample_count] = desc
        return sample

    @property
    def is_completed(self):
        """Whether to complete algorithm."""
        return self.sample_count >= self.random_samples + self.num_generation * self.num_individual

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.sample_count

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
            a_copy[k] = 1 - a_copy[k]
        return a_copy
