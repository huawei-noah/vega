# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import importlib
from modnas.registry.backend import build
from . import predefined

_backend = None

_backend_keys = []


def use(backend, *args, imported=False, **kwargs):
    """Switch to backend by name."""
    global _backend, _backend_keys
    if backend == _backend or backend in ['none', None]:
        return
    try:
        if imported:
            bk_mod = importlib.import_module(backend)
        else:
            bk_mod = build(backend, *args, **kwargs)
    except ImportError:
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
    _backend_keys = bk_keys
    _backend = backend


def backend():
    """Return name of current backend."""
    return _backend


def is_backend(backend):
    """Return if the current backend is the given one."""
    return _backend == backend
