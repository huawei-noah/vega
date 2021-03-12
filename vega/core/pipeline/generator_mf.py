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

from .generator import Generator


class GeneratorMF(Generator):
    """Convert search space and search algorithm, sample a new model."""

    def __init__(self):
        """Initialize an instance of the GeneratorMF class."""
        super(GeneratorMF, self).__init__()

    def sample(self):
        """Sample a work id and model from search algorithm."""
        iter_id, model_desc, epochs = self.search_alg.search()
        return (iter_id, model_desc, epochs)
