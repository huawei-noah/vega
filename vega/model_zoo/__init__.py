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

"""Import and register torch vision model automatically."""

from .model_zoo import ModelZoo
from .tuner import ModelTuner


def register_modelzoo(backend):
    """Import and register modelzoo automatically."""
    if backend != "pytorch":
        return
    from .torch_vision_model import import_all_torchvision_models
    import logging
    try:
        import_all_torchvision_models()
    except Exception as e:
        logging.warn("Failed to import torchvision models, msg={}".format(str(e)))
