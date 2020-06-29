# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""SearchAlgorithm."""
from vega.core.common.task_ops import TaskOps
from vega.core.common.class_factory import ClassFactory, ClassType


class SearchAlgorithm(TaskOps):
    """SearchAlgorithm the base class for user defined search algorithms.

    :param search_space: User defined `search_space`, default is None.
    :type search_space: SearchSpace
    :param **kwargs: `**kwargs`.
    :type **kwargs: type
    """

    def __new__(cls, search_space=None, **kwargs):
        """Create search algorithm instance by ClassFactory."""
        t_cls = ClassFactory.get_cls(ClassType.SEARCH_ALGORITHM)
        return super().__new__(t_cls)

    def __init__(self, search_space=None, **kwargs):
        """Init SearchAlgorithm."""
        super(SearchAlgorithm, self).__init__(self.cfg)
        self.policy = self.cfg.get('policy')
        self.range = self.cfg.get('range')

    def search(self):
        """Search function, Not Implemented Yet."""
        raise NotImplementedError

    def sample(self):
        """Sample function, Not Implemented Yet."""
        raise NotImplementedError

    def update(self, local_worker_path):
        """Update function, Not Implemented Yet.

        :param local_worker_path: the local path that saved `performance.txt`.
        :type local_worker_path: str
        """
        raise NotImplementedError

    def save_output(self, local_output_path):
        """Update function, Not Implemented Yet.

        :param local_output_path: the local output path to save the final results.
        :type local_output_path: str
        """
        return

    @property
    def is_completed(self):
        """If the search is finished, Not Implemented Yet."""
        return False
