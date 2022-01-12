# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
