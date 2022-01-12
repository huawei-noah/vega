# -*- coding: utf-8 -*-

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

"""Defined NetworkDesc."""
import logging
from copy import deepcopy
from vega.common import Config
from vega.common.class_factory import ClassFactory, ClassType


class NetworkDesc(object):
    """NetworkDesc."""

    def __init__(self, desc):
        """Init NetworkDesc."""
        self._desc = Config(deepcopy(desc))

    def to_model(self):
        """Transform a NetworkDesc to a special model."""
        logging.debug("Start to Create a Network.")
        module_type = self._desc.get('type', None)
        if module_type == "DagNetwork":
            module = ClassFactory.get_cls(ClassType.NETWORK, module_type)
        else:
            module = ClassFactory.get_cls(ClassType.NETWORK, "Module")
        model = module.from_desc(self._desc)
        if not model:
            raise Exception("Failed to create model, model desc={}".format(self._desc))
        model.desc = self._desc
        if hasattr(model, '_apply_names'):
            model._apply_names()
        return model
