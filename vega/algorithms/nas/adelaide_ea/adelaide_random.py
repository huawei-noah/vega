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

"""Random search algorithm for Adelaide EA."""
import random
from copy import deepcopy
from vega.common import ClassFactory, ClassType
from vega.core.search_algs import SearchAlgorithm
from .conf import AdelaideConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class AdelaideRandom(SearchAlgorithm):
    """Search algorithm of the random structures."""

    config = AdelaideConfig()

    def __init__(self, search_space=None):
        """Construct the AdelaideRandom class.

        :param search_space: Config of the search space
        """
        super(AdelaideRandom, self).__init__(search_space)
        self.search_space = search_space
        self.max_sample = self.config.max_sample
        self.sample_count = 0

    @property
    def is_completed(self):
        """Tell whether the search process is completed.

        :return: True is completed, or False otherwise
        """
        return self.sample_count >= self.max_sample

    def search(self):
        """Search one random model.

        :return: current number of samples, and the model
        """
        desc = deepcopy(self.search_space)
        search_desc = self.search_space.custom
        num_ops = len(search_desc.op_names)
        ops = [random.randrange(num_ops) for _ in range(7)]
        inputs = list()
        for inputs_index in range(3):
            for i in range(2):
                inputs.append(random.randint(0, (inputs_index + 2) * 3 - 5))
        conns = list()
        for conns_index in range(3):
            for i in range(2):
                conns.append(random.randint(0, conns_index + 3))
        decoder_cell_str = list()
        decoder_cell_str.append(ops[0])
        decoder_cell_str.append([inputs[0], inputs[1], ops[1], ops[2]])
        decoder_cell_str.append([inputs[2], inputs[3], ops[3], ops[4]])
        decoder_cell_str.append([inputs[4], inputs[5], ops[5], ops[6]])
        decoder_conn_str = [[conns[0], conns[1]], [conns[2], conns[3]], [conns[4], conns[5]]]
        decoder_arch_str = [decoder_cell_str, decoder_conn_str]
        search_desc['config'] = decoder_arch_str
        search_desc['method'] = "random"
        search_desc = self.codec.encode(search_desc)
        self.sample_count += 1
        desc['custom'] = search_desc
        return self.sample_count, desc

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.max_sample
