# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""VEGA Registry."""

from vega.common import ClassFactory, ClassType
from modnas.utils.logging import get_logger


def get_reg_type(type_name):
    """Return ClassType from registry path."""
    type_name = 'modnas.{}'.format(type_name)
    attr = type_name.upper()
    reg_type = getattr(ClassType, attr, None)
    if reg_type is None:
        setattr(ClassType, attr, None)
    return reg_type


class Registry():
    """Registry class."""

    logger = get_logger('registry')

    def __init__(self, allow_replace=False):
        self.allow_replace = allow_replace

    def get_reg_name(self, name):
        """Return proper registration name."""
        return name.lower().replace('-', '').replace('_', '').replace(' ', '')

    def register(self, regclass, reg_path, reg_id):
        """Register a component class."""
        reg_id = self.get_reg_name(reg_id)
        ClassFactory.register_cls(regclass, type_name=get_reg_type(reg_path), alias=reg_id)
        self.logger.debug('registered: {}'.format(reg_id))

    def get(self, reg_path, reg_id):
        """Return registered class by name."""
        reg_id = self.get_reg_name(reg_id)
        return ClassFactory.get_cls(get_reg_type(reg_path), reg_id)


registry = Registry()
