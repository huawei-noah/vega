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
