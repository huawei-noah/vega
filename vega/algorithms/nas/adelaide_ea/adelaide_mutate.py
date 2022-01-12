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

"""Random search algorithm for AdelaideEA."""
import logging
import random
from copy import deepcopy
from vega.common import ClassFactory, ClassType
from vega.report import ReportServer
from vega.core.search_algs import SearchAlgorithm
from .conf import AdelaideConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class AdelaideMutate(SearchAlgorithm):
    """Search algorithm of the random structures."""

    config = AdelaideConfig()

    def __init__(self, search_space=None):
        """Construct the AdelaideMutate class.

        :param search_space: Config of the search space
        """
        super(AdelaideMutate, self).__init__(search_space)
        self.max_sample = self.config.max_sample
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
        records = ReportServer().get_pareto_front_records(['random', 'mutate'])
        codes = []
        for record in records:
            custom = record.desc['custom']
            codes.append(custom['code'])
        num_ops = len(search_desc.op_names)
        upper_bounds = [num_ops, 2, 2, num_ops, num_ops, 5, 5, num_ops, num_ops,
                        8, 8, num_ops, num_ops, 4, 4, 5, 5, 6, 6]
        code_to_mutate = random.choice(codes)
        index = random.randrange(len(upper_bounds))
        choices = list(range(upper_bounds[index]))
        choices.pop(int(code_to_mutate[index + 1], 36))
        choice = random.choice(choices)
        code_mutated = code_to_mutate[:index + 1] + str(choice) + code_to_mutate[index + 2:]
        search_desc['code'] = code_mutated
        search_desc['method'] = "mutate"
        logging.info("Mutate from {} to {}".format(code_to_mutate, code_mutated))
        search_desc = self.codec.decode(search_desc)
        self.sample_count += 1
        desc['custom'] = search_desc
        return self.sample_count, desc

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.max_sample
