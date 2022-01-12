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

"""Registry for framework components."""
import sys
import types
import importlib.util
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence, Union
from types import ModuleType
from .registry import registry


SPEC_TYPE = Union[str, Tuple[str, ...], List[Any], Dict[str, Any]]


def register(_reg_path: str, builder: Any, _reg_id: Optional[str] = None) -> Any:
    """Register class as name."""
    if _reg_id is None:
        _reg_id = builder.__qualname__
    registry.register(builder, _reg_path, _reg_id)
    return builder


def get_builder(_reg_path: str, _reg_id: str) -> Any:
    """Return class builder by name."""
    return registry.get(_reg_path, _reg_id)


def parse_spec(spec: SPEC_TYPE) -> Any:
    """Return parsed id and arguments from build spec."""
    if isinstance(spec, dict):
        return spec['type'], spec.get('args', {})
    if isinstance(spec, (tuple, list)) and isinstance(spec[0], str):
        return spec[0], {} if len(spec) < 2 else spec[1]
    if isinstance(spec, str):
        return spec, {}
    raise ValueError('Invalid build spec: {}'.format(spec))


def to_spec(reg_id: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return build spec from id and arguments."""
    return {
        'type': reg_id,
        'args': kwargs
    }


def streamline_spec(spec: Optional[Union[Dict[str, SPEC_TYPE], List[SPEC_TYPE], SPEC_TYPE]]) -> List[SPEC_TYPE]:
    """Return a list of one or multiple specs."""
    if spec is None:
        return []
    if isinstance(spec, dict) and 'type' not in spec:
        return list(spec.values())
    if not isinstance(spec, list):
        return [spec]
    return spec


def build(_reg_path: str, _spec: SPEC_TYPE, *args, **kwargs) -> Any:
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


def get_registry_utils(_reg_path: str) -> Tuple[str, Callable, Callable, Callable, Callable]:
    """Return registration utilities."""
    _register = partial(register, _reg_path)
    _get_builder = partial(get_builder, _reg_path)
    _build = partial(build, _reg_path)
    _register_as = partial(register_as, _reg_path)
    return _reg_path, _register, _get_builder, _build, _register_as


def _get_registry_name(path: List[str]) -> str:
    return '.'.join(path[path.index('modnas') + 2:])


class RegistryModule(ModuleType):
    """Registry as a module."""

    def __init__(self, fullname: str) -> None:
        path = fullname.split('.')
        registry_name = _get_registry_name(path)
        self.__package__ = fullname
        self.__path__ = path
        self.__name__ = registry_name
        self.__loader__ = None
        self.__spec__ = None
        self.reg_path, self.register, self.get_builder, self.build, self.register_as = get_registry_utils(registry_name)

    def __getattr__(self, attr: str) -> Any:
        """Return builder by attribute name."""
        if attr in self.__dict__:
            return self.__dict__.get(attr)
        return self.get_builder(attr)


class RegistryImporter(Loader, MetaPathFinder):
    """Create new Registry using import hooks (PEP 302)."""

    def find_spec(
        self, fullname: str, path: Optional[Sequence[Union[bytes, str]]], target: Optional[types.ModuleType] = None
    ) -> Optional[ModuleSpec]:
        """Handle registry imports."""
        if 'modnas.registry' in fullname:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def load_module(self, fullname: str) -> RegistryModule:
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
