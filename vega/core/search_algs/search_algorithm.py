# -*- coding: utf-8 -*-

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

"""SearchAlgorithm."""
import logging
import json
import glob
from vega.common import TaskOps, FileOps
from vega.common import ClassFactory, ClassType
from vega.report import ReportRecord
from vega.core.search_algs.codec import Codec
from vega.core.pipeline.conf import PipeStepConfig


class SearchAlgorithm(TaskOps):
    """SearchAlgorithm the base class for user defined search algorithms.

    :param search_space: User defined `search_space`, default is None.
    :type search_space: SearchSpace
    :param **kwargs: `**kwargs`.
    :type **kwargs: type
    """

    config = None

    def __new__(cls, *args, **kwargs):
        """Create search algorithm instance by ClassFactory."""
        if cls.__name__ != 'SearchAlgorithm':
            return super().__new__(cls)
        if kwargs.get('type'):
            t_cls = ClassFactory.get_cls(ClassType.SEARCH_ALGORITHM, kwargs.pop('type'))
        else:
            t_cls = ClassFactory.get_cls(ClassType.SEARCH_ALGORITHM, PipeStepConfig.search_algorithm.type)

        return super().__new__(t_cls)

    def __init__(self, search_space=None, **kwargs):
        """Init SearchAlgorithm."""
        super(SearchAlgorithm, self).__init__()
        # modify config by kwargs, using local scope
        if self.config and kwargs:
            self.config.from_dict(kwargs)
        self.search_space = search_space
        if hasattr(self.config, 'codec'):
            self.codec = Codec(search_space, type=self.config.codec)
        else:
            self.codec = None
        logging.debug("Config=%s", self.config)
        self.record = ReportRecord()
        self.record.step_name = self.step_name
        self._get_search_space_list()

    def _get_search_space_list(self):
        """Get search space list from models folder."""
        models_folder = PipeStepConfig.pipe_step.get("models_folder")
        if not models_folder:
            self.search_space_list = None
            return
        self.search_space_list = []
        models_folder = models_folder.replace("{local_base_path}", TaskOps().local_base_path)
        pattern = FileOps.join_path(models_folder, "desc*.json")
        files = glob.glob(pattern)
        for file in files:
            with open(file) as f:
                self.search_space_list.append(json.load(f))

    def search(self):
        """Search function, Not Implemented Yet."""
        raise NotImplementedError

    def update(self, record):
        """Update function, Not Implemented Yet.

        :param record: record dict.
        """
        pass

    @property
    def is_completed(self):
        """If the search is finished, Not Implemented Yet."""
        raise NotImplementedError

    @property
    def max_samples(self):
        """Max samples in search algorithms."""
        return 1
