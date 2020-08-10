# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined RandomHpo class."""
from vega.algorithms.hpo.common import RandomSearch
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.algorithms.hpo.hpo_base import HPOBase
from .random_conf import RandomConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class RandomHpo(HPOBase):
    """An Hpo of Random, inherit from HpoGenerator."""

    config = RandomConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init RandomHpo."""
        super(RandomHpo, self).__init__(search_space, **kwargs)
        config_count = self.config.policy.total_epochs // self.config.policy.epochs_per_iter
        self.hpo = RandomSearch(self.hps, config_count, self.config.policy.epochs_per_iter)
