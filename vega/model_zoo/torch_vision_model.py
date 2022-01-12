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

"""Import all torchvision networks and models."""
from types import ModuleType
from torchvision import models as torchvision_models
from vega.common import ClassType, ClassFactory


def import_all_torchvision_models():
    """Import all torchvision networks and models."""

    def _register_models_from_current_module_scope(module):
        for _name in dir(module):
            if _name.startswith("_"):
                continue
            _cls = getattr(module, _name)
            if isinstance(_cls, ModuleType):
                continue
            if ClassFactory.is_exists(ClassType.NETWORK, 'torchvision_' + _cls.__name__):
                continue
            ClassFactory.register_cls(_cls, ClassType.NETWORK, alias='torchvision_' + _cls.__name__)

    _register_models_from_current_module_scope(torchvision_models)
    _register_models_from_current_module_scope(torchvision_models.segmentation)
    _register_models_from_current_module_scope(torchvision_models.detection)
    _register_models_from_current_module_scope(torchvision_models.video)
