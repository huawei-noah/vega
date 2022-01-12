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

"""Defined DnetNas."""
import logging
from vega.core.search_algs import SearchAlgorithm
from vega.core.search_algs import ParetoFront
from vega.common import ClassFactory, ClassType
from .conf import DblockNasConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class DblockNas(SearchAlgorithm):
    """DnetNas.

    :param search_space: input search_space
    :type: SeachSpace
    """

    config = DblockNasConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init DnetNas."""
        super(DblockNas, self).__init__(search_space, **kwargs)
        self.max_sample = self.config.range.max_sample
        self.min_sample = self.config.range.min_sample
        self.sample_count = 0
        logging.info("inited DblockNas")
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
            sample_desc = self.search_space.sample()
            sample = self.codec.encode(sample_desc)
            if not self.pareto_front._add_to_board(id=self.sample_count + 1,
                                                   config=sample):
                sample = None
        self.sample_count += 1
        sample_desc = self.codec.decode(sample)
        logging.info(f"sample: {sample_desc['network.backbone.encoding']}")
        return dict(worker_id=self.sample_count, encoded_desc=sample_desc)

    def update(self, record):
        """Use train and evaluate result to update algorithm.

        :param performance: performance value from trainer or evaluator
        """
        perf = record.get("rewards")
        worker_id = record.get("worker_id")
        logging.info("update performance={}".format(perf))
        self.pareto_front.add_pareto_score(worker_id, perf)

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.max_sample
