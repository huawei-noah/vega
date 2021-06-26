# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ModNasAlgorithm."""

from functools import partial
from vega.common import ConfigSerializable
from vega.common import ClassFactory, ClassType
from vega.core.search_algs import SearchAlgorithm
from modnas.registry.optim import build


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class ModNasAlgorithm(SearchAlgorithm):
    """ModularNAS search algorithm.

    :param search_space: Input search_space.
    :type search_space: SearchSpace
    """

    config = type('ModNasAlgorithmConfig', (ConfigSerializable, ), {"objective_keys": "accuracy"})()

    def __init__(self, search_space=None, **kwargs):
        super(ModNasAlgorithm, self).__init__(search_space, **kwargs)
        self.optim = None
        self.step = partial(self._lazy_optim_attr, 'step')
        self.next = partial(self._lazy_optim_attr, 'next')
        self.has_next = partial(self._lazy_optim_attr, 'has_next')
        self.sample_idx = -1

    def _lazy_optim_attr(self, attr, *args, **kwargs):
        self.init_optim()
        return getattr(self.optim, attr)(*args, **kwargs)

    def init_optim(self):
        """Initialize ModularNAS Optimizer."""
        if self.optim is not None:
            return
        config = getattr(self.config, 'optim', None)
        if config:
            self.set_optim(build(config))

    def set_optim(self, optim):
        """Set the current ModularNAS Optimizer."""
        self.optim = optim

    def search(self):
        """Search function."""
        self.sample_idx += 1
        return self.sample_idx, self.search_space

    @property
    def is_completed(self):
        """Check if the search is finished."""
        return self.sample_idx >= 0
