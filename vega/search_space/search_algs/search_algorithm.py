# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""SearchAlgorithm."""
import logging

from vega.core.common.config import obj2config
from vega.core.common.loader import load_conf_from_desc
from vega.core.common.task_ops import TaskOps
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.report import Report, ReportRecord
from vega.search_space.codec import Codec


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
            t_cls = ClassFactory.get_cls(ClassType.SEARCH_ALGORITHM)

        return super().__new__(t_cls)

    def __init__(self, search_space=None, **kwargs):
        """Init SearchAlgorithm."""
        super(SearchAlgorithm, self).__init__()
        # modify config by kwargs, using local scope
        if self.config and kwargs:
            self.config = self.config()
            load_conf_from_desc(self.config, kwargs)
        self.search_space = search_space
        if hasattr(self.config, 'codec'):
            self.codec = Codec(search_space, type=self.config.codec)
        else:
            self.codec = None
        logging.debug("Config=%s", obj2config(self.config))
        self.report = Report()
        self.record = ReportRecord()
        self.record.step_name = self.step_name

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
