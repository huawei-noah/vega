# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ArchDesc Constructors."""
import os
import yaml
import json
import copy
from .default import DefaultSlotTraversalConstructor
from modnas.registry.arch_space import build as build_module
from modnas.registry.construct import register
from modnas.arch_space.slot import Slot
from modnas.utils.logging import get_logger


logger = get_logger('construct')


_arch_desc_parser = {
    'json': lambda desc: json.loads(desc),
    'yaml': lambda desc: yaml.load(desc, Loader=yaml.SafeLoader),
    'yml': lambda desc: yaml.load(desc, Loader=yaml.SafeLoader),
}


def parse_arch_desc(desc, parser=None):
    """Return archdesc parsed from file."""
    if isinstance(desc, str):
        default_parser = 'yaml'
        if os.path.exists(desc):
            _, ext = os.path.splitext(desc)
            default_parser = ext[1:].lower()
            with open(desc, 'r', encoding='UTF-8') as f:
                desc = f.read()
        parser = parser or default_parser
        parse_fn = _arch_desc_parser.get(parser)
        if parse_fn is None:
            raise ValueError('invalid arch_desc parser type: {}'.format(parser))
        return parse_fn(desc)
    else:
        return desc


class DefaultArchDescConstructor():
    """Constructor that builds network from archdesc."""

    def __init__(self, arch_desc, parse_args=None):
        arch_desc = parse_arch_desc(arch_desc, **(parse_args or {}))
        logger.info('construct from arch_desc: {}'.format(arch_desc))
        self.arch_desc = arch_desc

    def __call__(self, *args, **kwargs):
        """Run constructor."""
        raise NotImplementedError


@register
class DefaultRecursiveArchDescConstructor(DefaultArchDescConstructor):
    """Constructor that recursively builds network submodules from archdesc."""

    def __init__(self, arch_desc, parse_args=None, construct_fn='build_from_arch_desc', fn_args=None, substitute=False):
        super().__init__(arch_desc, parse_args)
        self.construct_fn = construct_fn
        self.fn_args = fn_args or {}
        self.substitute = substitute

    def visit(self, module):
        """Construct and return module."""
        construct_fn = getattr(module, self.construct_fn, None)
        if construct_fn is not None:
            ret = construct_fn(self.arch_desc, **copy.deepcopy(self.fn_args))
            return module if ret is None else ret
        for n, m in module.named_children():
            m = self.visit(m)
            if m is not None and self.substitute:
                module.add_module(n, m)
        return module

    def __call__(self, model):
        """Run constructor."""
        Slot.set_convert_fn(self.convert)
        return self.visit(model)

    def convert(self, slot, desc, *args, **kwargs):
        """Convert Slot to module from archdesc."""
        desc = desc[0] if isinstance(desc, list) else desc
        return build_module(desc, slot, *args, **kwargs)


@register
class DefaultSlotArchDescConstructor(DefaultSlotTraversalConstructor, DefaultArchDescConstructor):
    """Constructor that converts Slots to modules from archdesc."""

    def __init__(self, arch_desc, parse_args=None, fn_args=None):
        DefaultSlotTraversalConstructor.__init__(self)
        DefaultArchDescConstructor.__init__(self, arch_desc, parse_args)
        self.fn_args = fn_args or {}
        self.idx = -1

    def get_next_desc(self):
        """Return next archdesc item."""
        self.idx += 1
        desc = self.arch_desc[self.idx]
        if isinstance(desc, list) and len(desc) == 1:
            desc = desc[0]
        return desc

    def convert(self, slot):
        """Convert Slot to module from archdesc."""
        m_type = self.get_next_desc()
        return build_module(m_type, slot, **copy.deepcopy(self.fn_args))
