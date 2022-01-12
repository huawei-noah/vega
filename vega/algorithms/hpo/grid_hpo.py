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

"""Defined GridSearch class."""
from vega.common import ClassFactory, ClassType
from .random_hpo import RandomSearch


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class GridSearch(RandomSearch):
    """An Hpo of GridSearch."""

    def __init__(self, search_space=None, **kwargs):
        """Init GridSearch."""
        super(GridSearch, self).__init__(search_space, **kwargs)
        self.sample_count = 0
        self.params = self.search_space.get_sample_space(gridding=True)
        self.max_sample = len(self.params)

    def search(self):
        """Search function, Not Implemented Yet."""
        param = self.search_space.decode(self.params[self.sample_count])
        self.sample_count += 1
        return {"worker_id": self.sample_count, "encoded_desc": param}
