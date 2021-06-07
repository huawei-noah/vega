# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Default registry."""
from modnas.utils.logging import get_logger


class Registry():
    """Registry class."""

    logger = get_logger('registry')

    def __init__(self, allow_replace=False):
        self.allow_replace = allow_replace
        self._reg_class = {}

    def get_full_path(self, reg_path, reg_id):
        """Return full registration path."""
        return '{}.{}'.format(reg_path, reg_id)

    def get_reg_name(self, reg_path, reg_id):
        """Return proper registration name."""
        name = self.get_full_path(reg_path, reg_id)
        return name.lower().replace('-', '').replace('_', '').replace(' ', '')

    def register(self, regclass, reg_path, reg_id):
        """Register a component class."""
        reg_id = self.get_reg_name(reg_path, reg_id)
        if reg_id in self._reg_class:
            self.logger.warning('re-register id: {}'.format(reg_id))
            if not self.allow_replace:
                raise ValueError('Cannot re-register id: {}'.format(reg_id))
        self._reg_class[reg_id] = regclass
        self.logger.debug('registered: {}'.format(reg_id))

    def get(self, reg_path, reg_id):
        """Return registered class by name."""
        reg_id = self.get_reg_name(reg_path, reg_id)
        if reg_id not in self._reg_class:
            raise ValueError('id \'{}\' not found in registry'.format(reg_id))
        return self._reg_class[reg_id]


registry = Registry()
