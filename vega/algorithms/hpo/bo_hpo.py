# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined BoHpo class."""
from vega.algorithms.hpo.common import BO
from zeus.common import ClassFactory, ClassType
from vega.algorithms.hpo.hpo_base import HPOBase
from .bo_conf import BoConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class BoHpo(HPOBase):
    """An Hpo of Bayesian Optimization, inherit from HpoGenerator."""

    config = BoConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init BoHpo."""
        super(BoHpo, self).__init__(search_space, **kwargs)
        config_count = self.config.policy.total_epochs // self.config.policy.max_epochs
        self.hpo = BO(self.search_space, config_count,
                      self.config.policy.max_epochs,
                      self.config.policy.warmup_count,
                      self.config.policy.alg_name)
