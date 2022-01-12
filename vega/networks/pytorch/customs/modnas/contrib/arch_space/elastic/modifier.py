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

"""Module states modifier."""
from typing import Callable, Union
from torch.nn.modules.module import Module
from torch import Tensor


def get_ori_param(module, name):
    """Return original module parameter."""
    return module._params_ori[name]


def get_ori_buffer(module: Module, name: str) -> Tensor:
    """Return original module buffer."""
    return module._buffers_ori[name]


def get_ori_attr(module, name):
    """Return original module attribute."""
    return module._attrs_ori[name]


def backup_param(module: Module, name: str) -> None:
    """Backup module parameter."""
    if not hasattr(module, '_params_ori'):
        module._params_ori = dict()
    if name in module._params_ori:
        return
    val = module._parameters[name]
    module._params_ori[name] = val


def backup_buffer(module: Module, name: str) -> None:
    """Backup module buffer."""
    if not hasattr(module, '_buffers_ori'):
        module._buffers_ori = dict()
    if name in module._buffers_ori:
        return
    val = module._buffers[name]
    module._buffers_ori[name] = val


def backup_attr(module: Module, name: str) -> None:
    """Backup module attribute."""
    if not hasattr(module, '_attrs_ori'):
        module._attrs_ori = dict()
    if name in module._attrs_ori:
        return
    val = getattr(module, name)
    module._attrs_ori[name] = val


def update_param(module, name, val):
    """Update module parameter."""
    if not hasattr(module, '_params_ori'):
        return
    if name not in module._params_ori:
        return
    module._params_ori[name] = val


def update_buffer(module, name, val):
    """Update module buffer."""
    if not hasattr(module, '_buffers_ori'):
        return
    if name not in module._buffers_ori:
        return
    module._buffers_ori[name] = val


def update_attr(module, name, val):
    """Update module attribute."""
    if not hasattr(module, '_attrs_ori'):
        return
    if name not in module._attrs_ori:
        return
    module._attrs_ori[name] = val


def restore_param(module, name):
    """Restore module parameter."""
    if not hasattr(module, '_params_ori'):
        return
    if name not in module._params_ori:
        return
    val = module._params_ori.pop(name)
    module._parameters[name] = val


def restore_buffer(module, name):
    """Restore module restore_buffer."""
    if not hasattr(module, '_buffers_ori'):
        return
    if name not in module._buffers_ori:
        return
    val = module._buffers_ori.pop(name)
    module._buffers[name] = val


def restore_attr(module, name):
    """Restore module attribute."""
    if not hasattr(module, '_attrs_ori'):
        return
    if name not in module._attrs_ori:
        return
    val = module._attrs_ori.pop(name)
    object.__setattr__(module, name, val)


def modify_param(module: Module, name: str, value: Tensor) -> None:
    """Modify module parameter."""
    backup_param(module, name)
    module._parameters[name] = value


def modify_buffer(module: Module, name: str, value: Tensor) -> None:
    """Modify module buffer."""
    backup_buffer(module, name)
    module._buffers[name] = value


def modify_attr(module: Module, name: str, value: Union[Callable, int]) -> None:
    """Modify module attribute."""
    backup_attr(module, name)
    object.__setattr__(module, name, value)


def restore_module_parameters(module: Module) -> None:
    """Restore module parameters."""
    if hasattr(module, '_params_ori'):
        module._parameters.update(module._params_ori)
        module._params_ori.clear()


def restore_module_buffers(module: Module) -> None:
    """Restore module buffers."""
    if hasattr(module, '_buffers_ori'):
        module._buffers.update(module._buffers_ori)
        module._buffers_ori.clear()


def restore_module_attrs(module: Module) -> None:
    """Restore module attributes."""
    if hasattr(module, '_attrs_ori'):
        module.__dict__.update(module._attrs_ori)
        module._attrs_ori.clear()


def restore_module_states(module: Module) -> None:
    """Restore all module states."""
    restore_module_parameters(module)
    restore_module_buffers(module)
    restore_module_attrs(module)
