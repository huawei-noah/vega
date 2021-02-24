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

from .generator import Generator


class GeneratorMF(Generator):
    """Convert search space and search algorithm, sample a new model."""

    def __init__(self):
        super(GeneratorMF, self).__init__()

    def sample(self):
        """Sample a work id and model from search algorithm."""
        id, params, epochs = self.search_alg.search()
        model = params.to_model()
        return id, model, epochs

