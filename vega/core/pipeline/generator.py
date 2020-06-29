# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Generator for NasPipeStep."""
import logging
from vega.search_space.search_algs import SearchAlgorithm
from vega.search_space.search_space import SearchSpace
from vega.search_space.networks.network_desc import NetworkDesc


class Generator(object):
    """Convert search space and search algorithm, sample a new model."""

    _subclasses = {}

    def __init__(self):
        self.search_space = SearchSpace()
        self.search_alg = SearchAlgorithm(self.search_space)

    @property
    def is_completed(self):
        """Define a property to determine search algorithm is completed."""
        return self.search_alg.is_completed

    def sample(self):
        """Sample a work id and model from search algorithm."""
        id, params = self.search_alg.search()
        if isinstance(params, NetworkDesc):
            model = params.to_model()
            return id, model
        elif isinstance(params, list):
            id_list = id
            model_list = []
            for param in params:
                model_list.append(param.to_model())
            return id_list, model_list
        if id is None or params is None:
            return None, None
        self.search_space.update(params)
        model = self.search_space.from_desc().to_model()
        return id, model

    def update(self, worker_path):
        """Update search algorithm accord to the worker path.

        :param worker_path: worker path of current task
        :return:
        """
        self.search_alg.update(worker_path)
