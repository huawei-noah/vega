# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from functools import partial
import importlib
from modnas.registry.backend import register


register(partial(importlib.import_module, 'modnas.backend.predefined.torch'), 'torch')
register(partial(importlib.import_module, 'modnas.backend.predefined.tensorflow'), 'tensorflow')
