# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined RandomSearch class."""
from vega.common import ClassFactory, ClassType
from vega.core.search_algs import SearchAlgorithm
from .random_conf import RandomConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class RandomSearch(SearchAlgorithm):
    """An Hpo of Random."""

    config = RandomConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init RandomSearch."""
        super(RandomSearch, self).__init__(search_space, **kwargs)
        self.sample_count = 0
        self.max_sample = self.config.policy.num_sample

    def search(self):
        """Search function, Not Implemented Yet."""
        self.sample_count += 1
        params = self.search_space.sample()
        return {"worker_id": self.sample_count, "encoded_desc": params}

    @property
    def is_completed(self):
        """Check is completed."""
        return self.sample_count >= self.max_sample

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.max_sample
