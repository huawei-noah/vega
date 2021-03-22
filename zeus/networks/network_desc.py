# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined NetworkDesc."""
import logging
from copy import deepcopy
from zeus.common import Config
from zeus.common.class_factory import ClassFactory, ClassType


class NetworkDesc(object):
    """NetworkDesc."""

    def __init__(self, desc, is_deformation=False):
        """Init NetworkDesc."""
        self._desc = Config(deepcopy(desc))
        self.is_deformation = is_deformation

    def to_model(self):
        """Transform a NetworkDesc to a special model."""
        logging.debug("Start to Create a Network.")
        module = ClassFactory.get_cls(ClassType.NETWORK, "Module")
        model = module.from_desc(self._desc)
        if not model:
            raise Exception("Failed to create model, model desc={}".format(self._desc))
        if not self.is_deformation:
            model.desc = self._desc
        return model
