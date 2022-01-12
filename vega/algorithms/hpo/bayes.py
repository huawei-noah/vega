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

"""Defined Bayes Search class."""
from vega.common import ClassFactory, ClassType
from vega.core.search_algs import SearchAlgorithm
from vega.algorithms.hpo.bayes_conf import BayesConfig
from vega.algorithms.hpo.ea.ga import GeneticAlgorithm
from vega.algorithms.hpo.sha_base.tuner import TunerBuilder


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class BayesSearch(SearchAlgorithm):
    """An Hpo of Bayes optimization."""

    config = BayesConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init BayesSearch."""
        super(BayesSearch, self).__init__(search_space, **kwargs)
        self.num_samples = self.config.num_samples
        self._all_desc_dict = {}
        multi_obj = isinstance(self.config.objective_keys, list) and len(self.config.objective_keys) > 1
        alg_name = "GA" if multi_obj else self.config.tuner
        if alg_name == "GA":
            self.tuner = GeneticAlgorithm(search_space, random_samples=self.config.warmup_count,
                                          prob_crossover=self.config.prob_crossover,
                                          prob_mutatation=self.config.prob_mutatation)
        else:
            self.tuner = TunerBuilder(search_space=search_space, tuner=alg_name)
        self.sample_count = 0

    def search(self, config_id=None):
        """Search one NetworkDesc from search space."""
        desc = self.tuner.propose()[0]
        self.sample_count += 1
        self._all_desc_dict[str(self.sample_count)] = desc
        return dict(worker_id=self.sample_count, encoded_desc=desc)

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.num_samples

    @property
    def is_completed(self):
        """Whether to complete algorithm."""
        return self.sample_count >= self.num_samples

    def update(self, record):
        """Update function, Not Implemented Yet.

        :param record: record dict.
        """
        desc = self._all_desc_dict.get(str(record.get('worker_id')))
        rewards = record.get("rewards")
        self.tuner.add(desc, rewards)
