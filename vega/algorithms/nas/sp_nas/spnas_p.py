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

"""The second stage of SP-NAS."""

import logging
import random
import numpy as np
from vega.common import ClassFactory, ClassType
from vega.core.search_algs import SearchAlgorithm
from vega.report import ReportServer
from .conf import SpNasConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SpNasP(SearchAlgorithm):
    """Second Stage of SMNAS."""

    config = SpNasConfig()

    def __init__(self, search_space=None):
        super(SpNasP, self).__init__(search_space)
        self.sample_count = 0
        self.max_sample = self.config.max_sample

    @property
    def is_completed(self):
        """Check sampling if finished."""
        return self.sample_count > self.max_sample

    def search(self):
        """Search a sample."""
        pareto_records = ReportServer().get_pareto_front_records(choice='normal')
        best_record = pareto_records[0] if pareto_records else None
        desc = self.search_space.sample()
        if best_record and best_record.desc:
            desc['network.neck.code'] = self._mutate_parallelnet(best_record.desc.get("neck").get('code'))
        self.sample_count += 1
        if self.search_space_list:
            backbone = random.choice(self.search_space_list)
            desc['network.backbone.type'] = backbone.get("backbone").get('type')
            desc['network.backbone.code'] = backbone.get("backbone").get('code')
            desc['network.backbone.weight_file'] = backbone.get("backbone").get('weight_file')
        logging.info("desc:{}".format(desc))
        return dict(worker_id=self.sample_count, encoded_desc=desc)

    @property
    def max_samples(self):
        """Return the max number of samples."""
        return self.max_sample

    def _mutate_parallelnet(self, code):
        """Mutate operation in Parallel-level searching.

        :param code: base arch encode
        :type code: list
        :return: parallel arch encode after mutate
        :rtype: list
        """
        p = [0.4, 0.3, 0.2, 0.1]
        num_stage = len(code)
        return list(np.random.choice(4, size=num_stage, replace=True, p=p))
