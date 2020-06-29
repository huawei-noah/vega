# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a SearchAlgorithm for resnet example."""
import random
import logging
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.search_algs import SearchAlgorithm


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class RandomSearch(SearchAlgorithm):
    """Random Search."""

    def __init__(self, search_space):
        super(RandomSearch, self).__init__(search_space)
        self.hyper_parameters = search_space.cfg.get('hyper_parameters') or {}
        self.count = 0

    def search(self):
        """Search a params."""
        self.count += 1
        params = {}
        for param_key, param_values in self.hyper_parameters.items():
            params[param_key] = random.choice(param_values)
        logging.info("params:%s", params)
        return self.count, params

    def update(self, local_worker_path):
        """Update."""
        pass

    def update_params(self, params):
        """Update params into search_space."""
        return None

    @property
    def is_completed(self):
        """Make trail completed."""
        return self.count > 2
