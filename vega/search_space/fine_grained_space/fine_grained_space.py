# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Fine Grained SearchSpace Define."""
import logging
from collections import OrderedDict
from copy import deepcopy

from vega.core.common.class_factory import ClassFactory, ClassType
import torch.nn as nn
from vega.core.common.utils import update_dict_with_flatten_keys


class FineGrainedMeta(object):
    """Meta class of FineGrainedSpace."""

    def __init__(self, args, kwargs):
        self._scope = OrderedDict()
        self._params = args or kwargs

    def __setattr__(self, name, value):
        """Set attr into scope for search space.

        :param name: instance name
        :param value: class
        """
        if not name.startswith("_"):
            self._scope[name] = value
        self.__dict__[name] = value

    def add(self, name, value):
        """Add class into scope."""
        setattr(self, name, value)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class FineGrainedSpace(FineGrainedMeta):
    """FineGrainedSpace."""

    def __init__(self, *args, **kwargs):
        super(FineGrainedSpace, self).__init__(args, kwargs)
        self.constructor(*args, **kwargs)

    def constructor(self, *args, **kwargs):
        """Create SearchSpace interface."""
        pass

    def update(self, params):
        """Update params to SearchSpace."""
        update_dict_with_flatten_keys(self.cfg, params)

    @classmethod
    def from_desc(self):
        """Create object from desc."""
        return FineGrainedSpaceFactory.from_desc(self.cfg)

    def to_model(self):
        """Transform a NetworkDesc to a special model."""
        seq = OrderedDict()
        after_model_handler = []
        for module_name, module in self._scope.items():
            if isinstance(module, FineGrainedSpace):
                module = module.to_model()
            if module_name == 'process':
                after_model_handler.append(module)
            else:
                seq[module_name] = module
        if not seq:
            logging.debug("Search Space Sequential is None.")
            return None
        model = nn.Sequential(seq)
        for handler in after_model_handler:
            model = handler(model)
        return model


class FineGrainedSpaceFactory(object):
    """Factory to create a new space."""

    @staticmethod
    def from_desc(desc):
        """Create search space from desc."""
        desc = deepcopy(desc)
        modules = desc.get('modules')
        if not modules:
            return FineGrainedSpaceFactory.create_search_space(desc)
        search_space = FineGrainedSpace()
        for modules_name in modules:
            module_desc = desc.get(modules_name)
            module = FineGrainedSpaceFactory.create_search_space(module_desc)
            search_space.add(modules_name, module)
        return search_space

    @staticmethod
    def create_search_space(desc):
        """Create one search space from desc."""
        param = deepcopy(desc)
        module_type = param.pop('type')
        if module_type == 'FineGrainedSpace':
            return FineGrainedSpaceFactory.from_desc(param)
        module = ClassFactory.get_cls(ClassType.SEARCH_SPACE, module_type)
        return module(**param) if param else module()
