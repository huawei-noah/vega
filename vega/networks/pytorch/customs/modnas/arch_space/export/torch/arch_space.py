# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Torch Architecture Exporters."""
import copy
from modnas.arch_space.slot import Slot
from modnas.arch_space.mixed_ops import MixedOp
from modnas.registry.export import register


@register
class DefaultSlotTraversalExporter():
    """Exporter that outputs parameter values."""

    def __init__(self, export_fn='to_arch_desc', fn_args=None, gen=None):
        self.gen = gen
        self.export_fn = export_fn
        self.fn_args = fn_args or {}
        self.visited = set()

    def export(self, slot, *args, **kwargs):
        """Return exported archdesc from Slot."""
        if slot in self.visited:
            return None
        self.visited.add(slot)
        export_fn = getattr(slot.get_entity(), self.export_fn, None)
        return None if export_fn is None else export_fn(*args, **kwargs)

    def __call__(self, model):
        """Run Exporter."""
        if model is None:
            return
        Slot.set_export_fn(self.export)
        arch_desc = []
        gen = self.gen or Slot.gen_slots_model(model)
        for m in gen():
            if m in self.visited:
                continue
            arch_desc.append(m.to_arch_desc(**copy.deepcopy(self.fn_args)))
        self.visited.clear()
        return arch_desc


@register
class DefaultRecursiveExporter():
    """Exporter that recursively outputs archdesc of submodules."""

    def __init__(self, export_fn='to_arch_desc', fn_args=None):
        self.fn_args = fn_args or {}
        self.export_fn = export_fn
        self.visited = set()

    def export(self, slot, *args, **kwargs):
        """Return exported archdesc from Slot."""
        export_fn = getattr(slot.get_entity(), self.export_fn, None)
        return None if export_fn is None else export_fn(*args, **kwargs)

    def visit(self, module):
        """Return exported archdesc from current module."""
        if module in self.visited:
            return None
        self.visited.add(module)
        export_fn = getattr(module, self.export_fn, None)
        if export_fn is not None:
            ret = export_fn(**copy.deepcopy(self.fn_args))
            if ret is not None:
                return ret
        return {n: self.visit(m) for n, m in module.named_children()}

    def __call__(self, model):
        """Run Exporter."""
        Slot.set_export_fn(self.export)
        desc = self.visit(model)
        self.visited.clear()
        return desc


@register
class DefaultMixedOpExporter():
    """Exporter that outputs archdesc from mixed operators."""

    def __init__(self, fn_args=None):
        self.fn_args = fn_args or {}

    def __call__(self, model):
        """Run Exporter."""
        desc = [m.to_arch_desc(**self.fn_args) for m in MixedOp.gen(model)]
        return desc
