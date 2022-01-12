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
