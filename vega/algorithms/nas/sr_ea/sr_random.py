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
from copy import deepcopy
from vega.core.search_algs import SearchAlgorithm
from vega.common import ClassFactory, ClassType
from .conf import SRConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SRRandom(SearchAlgorithm):
    """Search algorithm of the random structures."""

    config = SRConfig()

    def __init__(self, search_space=None, **kwargs):
        """Construct the SRRandom class.

        :param search_space: Config of the search space
        """
        super(SRRandom, self).__init__(search_space, **kwargs)
        self.max_sample = self.config.policy.num_sample
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
        search_desc = desc.custom
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
        desc['custom'] = search_desc
        self.sample_count += 1
        return dict(worker_id=self.sample_count, encoded_desc=desc)

    def update(self, record):
        """Nothing need to update."""
        pass

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.max_sample
