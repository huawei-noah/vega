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

"""Mutate part of SR_EA algorithm."""
import logging
import random
from copy import deepcopy
from vega.common import ClassFactory, ClassType
from vega.report import ReportServer
from vega.core.search_algs import SearchAlgorithm
from .conf import SRConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SRMutate(SearchAlgorithm):
    """Search algorithm of the mutated structures."""

    config = SRConfig()

    def __init__(self, search_space=None, **kwargs):
        """Construct the class SRMutate.

        :param search_space: Config of the search space
        """
        super(SRMutate, self).__init__(search_space, **kwargs)
        self.max_sample = self.config.policy.num_sample
        self.num_mutate = self.config.policy.num_mutate
        self.sample_count = 0

    @property
    def is_completed(self):
        """Tell whether the search process is completed.

        :return: True is completed, or False otherwise
        """
        return self.sample_count >= self.max_sample

    def search(self):
        """Search one mutated model.

        :return: current number of samples, and the model
        """
        desc = deepcopy(self.search_space)
        search_desc = desc.custom
        # TODO: merge sr ea in one pipe step.
        records = ReportServer().get_pareto_front_records(['random', 'mutate'])
        codes = []
        for record in records:
            codes.append(record.desc['custom']['code'])
        code_to_mutate = random.choice(codes)
        current_mutate, code_mutated = 0, code_to_mutate
        num_candidates = len(search_desc["candidates"])
        while current_mutate < self.num_mutate:
            code_new = self.mutate_once(code_mutated, num_candidates)
            if code_new != code_mutated:
                current_mutate += 1
                code_mutated = code_new
        logging.info("Mutate from {} to {}".format(code_to_mutate, code_mutated))
        search_desc['code'] = code_mutated
        search_desc['method'] = "mutate"
        search_desc = self.codec.decode(search_desc)
        desc['custom'] = search_desc
        self.sample_count += 1
        return dict(worker_id=self.sample_count, encoded_desc=desc)

    def mutate_once(self, code, num_largest):
        """Do one mutate.

        :param code: original code
        :param num_largest: number of candidates (largest number in code)
        :return: the mutated code
        """
        fun = random.choice([self.flip_once, self.insert_once, self.erase, self.swap_once])
        return fun(code, num_largest)

    @staticmethod
    def flip_once(code, num_largest):
        """Flip one block.

        :param code: original code
        :param num_largest: number of candidates (largest number in code)
        :return: the mutated code
        """
        index_to_flip = random.choice([index for index in range(len(code)) if code[index] != '+'])
        flip_choices = list(map(str, range(num_largest)))
        flip_choices.remove(code[index_to_flip])
        ch_flipped = random.choice(flip_choices)
        return code[:index_to_flip] + ch_flipped + code[index_to_flip + 1:]

    @staticmethod
    def insert_once(code, num_largest):
        """Insert one block.

        :param code: original code
        :param num_largest: number of candidates (largest number in code)
        :return: the mutated code
        """
        ch_insert = random.choice(list(map(str, range(num_largest))))
        place_insert = random.randint(0, len(code))
        return code[:place_insert] + ch_insert + code[place_insert:]

    @staticmethod
    def erase(code, num_largest):
        """Erase one block.

        :param code: original code
        :param num_largest: number of candidates (largest number in code)
        :return: the mutated code
        """
        place_choices, index = list(), 0
        while index < len(code):
            if code[index] == '+':
                index += 3
            else:
                place_choices.append(index)
                index += 1
        if len(place_choices) == 0:
            return code
        place_chosen = random.choice(place_choices)
        return code[:place_chosen] + code[place_chosen + 1:]

    @staticmethod
    def swap_once(code, num_largest):
        """Swap two adjacent blocks.

        :param code: original code
        :param num_largest: number of candidates (largest number in code)
        :return: the mutated code
        """
        parts, index = list(), 0
        while index < len(code):
            if code[index] == '+':
                parts.append(code[index: index + 3])
                index += 3
            else:
                parts.append(code[index])
                index += 1
        if len(parts) < 2:
            return code
        valid_choices = [index for index in range(len(parts) - 2) if parts[index] != parts[index + 1]]
        if len(valid_choices) == 0:
            return code
        place_chosen = random.choice(valid_choices)
        parts[place_chosen], parts[place_chosen + 1] = parts[place_chosen + 1], parts[place_chosen]
        return ''.join(parts)

    def update(self, record):
        """Nothing need to update."""
        pass

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.max_sample
