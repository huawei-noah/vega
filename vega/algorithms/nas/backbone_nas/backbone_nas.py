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

"""Defined BackboneNas."""
import random
import logging
import numpy as np
from vega.core.search_algs import SearchAlgorithm
from vega.core.search_algs import ParetoFront
from vega.common import ClassFactory, ClassType
from .conf import BackboneNasConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class BackboneNas(SearchAlgorithm):
    """BackboneNas.

    :param search_space: input search_space
    :type: SeachSpace
    """

    config = BackboneNasConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init BackboneNas."""
        super(BackboneNas, self).__init__(search_space, **kwargs)
        # ea or random
        self.num_mutate = self.config.policy.num_mutate
        self.random_ratio = self.config.policy.random_ratio
        self.max_sample = self.config.range.max_sample
        self.min_sample = self.config.range.min_sample
        self.sample_count = 0
        logging.info("inited BackboneNas")
        self.pareto_front = ParetoFront(
            self.config.pareto.object_count, self.config.pareto.max_object_ids)
        self._best_desc_file = 'nas_model_desc.json'

    @property
    def is_completed(self):
        """Check if NAS is finished."""
        return self.sample_count > self.max_sample

    def search(self):
        """Search in search_space and return a sample."""
        sample = {}
        while sample is None or 'code' not in sample:
            pareto_dict = self.pareto_front.get_pareto_front()
            pareto_list = list(pareto_dict.values())
            if self.pareto_front.size < self.min_sample or random.random() < self.random_ratio or len(
                    pareto_list) == 0:
                sample_desc = self.search_space.sample()
                sample = self.codec.encode(sample_desc)
            else:
                sample = pareto_list[0]
            if sample is not None and 'code' in sample:
                code = sample['code']
                code = self.ea_sample(code)
                sample['code'] = code
            if not self.pareto_front._add_to_board(id=self.sample_count + 1,
                                                   config=sample):
                sample = None
        self.sample_count += 1
        logging.info(sample)
        sample_desc = self.codec.decode(sample)
        print(sample_desc)
        return dict(worker_id=self.sample_count, encoded_desc=sample_desc)

    def random_sample(self):
        """Random sample from search_space."""
        sample_desc = self.search_space.sample()
        sample = self.codec.encode(sample_desc, is_random=True)
        return sample

    def ea_sample(self, code):
        """Use EA op to change a arch code.

        :param code: list of code for arch
        :type code: list
        :return: changed code
        :rtype: list
        """
        new_arch = code.copy()
        self._insert(new_arch)
        self._remove(new_arch)
        self._swap(new_arch[0], self.num_mutate // 2)
        self._swap(new_arch[1], self.num_mutate // 2)
        return new_arch

    def update(self, record):
        """Use train and evaluate result to update algorithm.

        :param performance: performance value from trainer or evaluator
        """
        perf = record.get("original_rewards")
        worker_id = record.get("worker_id")
        logging.info("update performance={}".format(perf))
        self.pareto_front.add_pareto_score(worker_id, perf)

    def _insert(self, arch):
        """Random insert to arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        idx = np.random.randint(low=0, high=len(arch[0]))
        arch[0].insert(idx, 1)
        idx = np.random.randint(low=0, high=len(arch[1]))
        arch[1].insert(idx, 1)
        return arch

    def _remove(self, arch):
        """Random remove one from arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        # random pop arch[0]
        ones_index = [i for i, char in enumerate(arch[0]) if char == 1]
        idx = random.choice(ones_index)
        arch[0].pop(idx)
        # random pop arch[1]
        ones_index = [i for i, char in enumerate(arch[1]) if char == 1]
        idx = random.choice(ones_index)
        arch[1].pop(idx)
        return arch

    def _swap(self, arch, R):
        """Random swap one in arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        while True:
            not_ones_index = [i for i, char in enumerate(arch) if char != 1]
            idx = random.choice(not_ones_index)
            r = random.randint(1, R)
            direction = -r if random.random() > 0.5 else r
            try:
                arch[idx], arch[idx + direction] = arch[idx + direction], arch[
                    idx]
                break
            except Exception:
                logging.debug("Arch is not match, continue.")
                continue
        return arch

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.max_sample
