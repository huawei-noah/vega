# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined RandomParetoHpo class."""
from vega.algorithms.hpo.common import RandomPareto
from zeus.common import ClassFactory, ClassType
from vega.algorithms.hpo.hpo_base import HPOBase
from .random_pareto_conf import RandomParetoConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class RandomParetoHpo(HPOBase):
    """An Hpo of RandomPareto, inherit from HpoGenerator."""

    config = RandomParetoConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init RandomParetoHpo."""
        super(RandomParetoHpo, self).__init__(search_space, **kwargs)
        config_count = self.config.policy.total_epochs // self.config.policy.max_epochs
        self.hpo = RandomPareto(self.search_space,
                                config_count,
                                self.config.policy.max_epochs,
                                object_count=int(self.config.policy.pareto.object_count),
                                max_object_ids=self.config.policy.pareto.max_object_ids
                                )
