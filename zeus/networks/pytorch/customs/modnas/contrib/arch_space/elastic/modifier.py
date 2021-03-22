# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Module states modifier."""


def get_ori_param(module, name):
    """Return original module parameter."""
    return module._params_ori[name]


def get_ori_buffer(module, name):
    """Return original module buffer."""
    return module._buffers_ori[name]


def get_ori_attr(module, name):
    """Return original module attribute."""
    return module._attrs_ori[name]


def backup_param(module, name):
    """Backup module parameter."""
    if not hasattr(module, '_params_ori'):
        module._params_ori = dict()
    if name in module._params_ori:
        return
    val = module._parameters[name]
    module._params_ori[name] = val


def backup_buffer(module, name):
    """Backup module buffer."""
    if not hasattr(module, '_buffers_ori'):
        module._buffers_ori = dict()
    if name in module._buffers_ori:
        return
    val = module._buffers[name]
    module._buffers_ori[name] = val


def backup_attr(module, name):
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


def modify_param(module, name, value):
    """Modify module parameter."""
    backup_param(module, name)
    module._parameters[name] = value


def modify_buffer(module, name, value):
    """Modify module buffer."""
    backup_buffer(module, name)
    module._buffers[name] = value


def modify_attr(module, name, value):
    """Modify module attribute."""
    backup_attr(module, name)
    object.__setattr__(module, name, value)


def restore_module_parameters(module):
    """Restore module parameters."""
    if hasattr(module, '_params_ori'):
        module._parameters.update(module._params_ori)
        module._params_ori.clear()


def restore_module_buffers(module):
    """Restore module buffers."""
    if hasattr(module, '_buffers_ori'):
        module._buffers.update(module._buffers_ori)
        module._buffers_ori.clear()


def restore_module_attrs(module):
    """Restore module attributes."""
    if hasattr(module, '_attrs_ori'):
        module.__dict__.update(module._attrs_ori)
        module._attrs_ori.clear()


def restore_module_states(module):
    """Restore all module states."""
    restore_module_parameters(module)
    restore_module_buffers(module)
    restore_module_attrs(module)
