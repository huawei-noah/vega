# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Registry for framework components."""
import sys
import importlib.util
from functools import partial
from .registry import registry


def register(_reg_path, builder, _reg_id=None):
    """Register class as name."""
    if _reg_id is None:
        _reg_id = builder.__qualname__
    registry.register(builder, _reg_path, _reg_id)
    return builder


def get_builder(_reg_path, _reg_id):
    """Return class builder by name."""
    return registry.get(_reg_path, _reg_id)


def parse_spec(spec):
    """Return parsed id and arguments from build spec."""
    if isinstance(spec, dict):
        return spec['type'], spec.get('args', {})
    if isinstance(spec, (tuple, list)) and isinstance(spec[0], str):
        return spec[0], {} if len(spec) < 2 else spec[1]
    if isinstance(spec, str):
        return spec, {}
    raise ValueError('Invalid build spec: {}'.format(spec))


def to_spec(reg_id, kwargs):
    """Return build spec from id and arguments."""
    return {
        'type': reg_id,
        'args': kwargs
    }


def build(_reg_path, _spec, *args, **kwargs):
    """Instantiate class by name."""
    reg_id, sp_kwargs = parse_spec(_spec)
    kwargs.update(sp_kwargs)
    return registry.get(_reg_path, reg_id)(*args, **kwargs)


def register_as(_reg_path, _reg_id=None):
    """Return a registration decorator."""
    def reg_builder(func):
        register(_reg_path, func, _reg_id)
        return func

    return reg_builder


def get_registry_utils(_reg_path):
    """Return registration utilities."""
    _register = partial(register, _reg_path)
    _get_builder = partial(get_builder, _reg_path)
    _build = partial(build, _reg_path)
    _register_as = partial(register_as, _reg_path)
    return _reg_path, _register, _get_builder, _build, _register_as


def _get_registry_name(path):
    return '.'.join(path[path.index('modnas') + 2:])


class RegistryModule():
    """Registry as a module."""

    def __init__(self, fullname):
        path = fullname.split('.')
        registry_name = _get_registry_name(path)
        self.__package__ = fullname
        self.__path__ = path
        self.__name__ = registry_name
        self.__loader__ = None
        self.__spec__ = None
        self.reg_path, self.register, self.get_builder, self.build, self.register_as = get_registry_utils(registry_name)

    def __getattr__(self, attr):
        """Return builder by attribute name."""
        if attr in self.__dict__:
            return self.__dict__.get(attr)
        return self.get_builder(attr)


class RegistryImporter():
    """Create new Registry using import hooks (PEP 302)."""

    def find_spec(self, fullname, path, target=None):
        """Handle registry imports."""
        if 'modnas.registry' in fullname:
            return importlib.util.spec_from_loader(fullname, self)

    def load_module(self, fullname):
        """Create and find registry by import path."""
        path = fullname.split('.')
        reg_path, reg_id = path[:-1], path[-1]
        reg_fullname = '.'.join(reg_path)
        registry_name = _get_registry_name(reg_path)
        if reg_fullname in sys.modules and len(registry_name):
            mod = get_builder(registry_name, reg_id)
            sys.modules[fullname] = mod
            return mod
        mod = sys.modules.get(fullname, RegistryModule(fullname))
        sys.modules[fullname] = mod
        return mod


sys.meta_path.append(RegistryImporter())
