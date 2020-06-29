# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Random search algorithm for SR_EA."""
import random
from vega.search_space.search_algs.search_algorithm import SearchAlgorithm
from vega.search_space.codec import Codec
from vega.search_space.networks import NetworkDesc
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SRRandom(SearchAlgorithm):
    """Search algorithm of the random structures."""

    def __init__(self, search_space=None):
        """Construct the SRRandom class.

        :param search_space: Config of the search space
        """
        super(SRRandom, self).__init__(search_space)
        self.search_space = search_space
        self.codec = Codec(self.cfg.codec, search_space)
        self.max_sample = self.policy.num_sample
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
        search_desc = self.search_space.search_space.custom
        num_blocks = random.randint(*search_desc.block_range)
        num_cibs = random.randint(*search_desc.cib_range)
        candidates = search_desc.candidates
        blocks = [random.choice(candidates) for _ in range(num_blocks)]
        for _ in range(num_cibs):
            cib = [random.choice(candidates) for _ in range(2)]
            blocks.insert(random.randint(0, len(blocks)), cib)
        search_desc['blocks'] = blocks
        search_desc['method'] = "random"
        search_desc = self.codec.encode(search_desc)
        self.sample_count += 1
        return self.sample_count, NetworkDesc(self.search_space.search_space)

    def update(self, local_worker_path):
        """Update function.

        :param local_worker_path: Local path that saved `performance.txt`
        :type local_worker_path: str
        """
        pass
