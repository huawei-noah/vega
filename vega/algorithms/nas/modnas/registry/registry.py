# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Default registry."""
import logging
from typing import Any

logger = logging.getLogger('modnas.registry')


class Registry():
    """Registry class."""

    def __init__(self, allow_replace: bool = False) -> None:
        self.allow_replace = allow_replace
        self._reg_class = {}

    def get_full_path(self, reg_path: str, reg_id: str) -> str:
        """Return full registration path."""
        return '{}.{}'.format(reg_path, reg_id)

    def get_reg_name(self, reg_path: str, reg_id: str) -> str:
        """Return proper registration name."""
        name = self.get_full_path(reg_path, reg_id)
        return name.lower().replace('-', '').replace('_', '').replace(' ', '')

    def register(self, regclass: Any, reg_path: str, reg_id: str) -> None:
        """Register a component class."""
        reg_id = self.get_reg_name(reg_path, reg_id)
        if reg_id in self._reg_class:
            logger.warning('re-register id: {}'.format(reg_id))
            if not self.allow_replace:
                raise ValueError('Cannot re-register id: {}'.format(reg_id))
        self._reg_class[reg_id] = regclass
        logger.debug('registered: {}'.format(reg_id))

    def get(self, reg_path: str, reg_id: str) -> Any:
        """Return registered class by name."""
        reg_id = self.get_reg_name(reg_path, reg_id)
        if reg_id not in self._reg_class:
            raise ValueError('id \'{}\' not found in registry'.format(reg_id))
        return self._reg_class[reg_id]


registry = Registry()
