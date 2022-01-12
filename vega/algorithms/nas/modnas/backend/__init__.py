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

import importlib
import logging
import traceback
from typing import Optional
from modnas.registry.backend import build
from . import predefined

_backend = None

_backend_keys = []


def use(backend_type: Optional[str], *args, imported=False, **kwargs) -> None:
    """Switch to backend by name."""
    global _backend, _backend_keys
    if backend_type == _backend or backend_type == 'none' or backend_type is None:
        return
    try:
        if imported:
            bk_mod = importlib.import_module(backend_type)
        else:
            bk_mod = build(backend_type, *args, **kwargs)
    except ImportError as e:
        logging.debug(traceback.format_exc())
        logging.error(f"error occured, message: {e}")
        return
    bk_vars = vars(bk_mod)
    bk_keys = bk_vars.keys()
    ns = globals()
    for k in _backend_keys:
        ns.pop(k, None)
    for k in bk_keys:
        if k.startswith('__'):
            continue
        ns[k] = bk_vars[k]
    _backend_keys = list(bk_keys)
    _backend = backend_type


def backend():
    """Return name of current backend."""
    return _backend


def is_backend(backend_type: str) -> bool:
    """Return if the current backend is the given one."""
    return _backend == backend_type
