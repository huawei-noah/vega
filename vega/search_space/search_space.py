# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Coarse SearchSpace Define."""
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.SEARCH_SPACE)
class SearchSpace(object):
    """Used for coarse search space. search space is the config from yaml."""

    def __new__(cls, *args, **kwargs):
        """Create a new SearchSpace."""
        t_cls = ClassFactory.get_cls(ClassType.SEARCH_SPACE)
        return super(SearchSpace, cls).__new__(t_cls)

    @property
    def search_space(self):
        """Get hyper parameters."""
        return self.cfg
