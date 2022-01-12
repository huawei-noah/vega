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

"""ArchDesc Constructors."""
import os
import json
import copy
from typing import Dict, Optional, Any, Sequence
import yaml
from torch.nn.modules.module import Module
from modnas.registry.arch_space import build as build_module
from modnas.registry.construct import register
from modnas.arch_space.slot import Slot
from modnas.utils.logging import get_logger
from vega.security.args import path_verify
from .default import DefaultSlotTraversalConstructor

logger = get_logger('construct')


_arch_desc_parser = {
    'json': lambda desc: json.loads(desc),
    'yaml': lambda desc: yaml.load(desc, Loader=yaml.SafeLoader),
    'yml': lambda desc: yaml.load(desc, Loader=yaml.SafeLoader),
}


def parse_arch_desc(desc: Any, parser: Optional[str] = None) -> Any:
    """Return archdesc parsed from file."""
    if isinstance(desc, str):
        default_parser = 'yaml'
        if os.path.exists(desc):
            desc = os.path.realpath(desc)
            desc = path_verify(desc)
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

    def __init__(self, arch_desc: Any, parse_args: Optional[Dict[str, Any]] = None) -> None:
        arch_desc = parse_arch_desc(arch_desc, **(parse_args or {}))
        logger.info('construct from arch_desc: {}'.format(arch_desc))
        self.arch_desc = arch_desc

    def __call__(self, *args, **kwargs):
        """Run constructor."""
        raise NotImplementedError


@register
class DefaultRecursiveArchDescConstructor(DefaultArchDescConstructor):
    """Constructor that recursively builds network submodules from archdesc."""

    def __init__(
        self, arch_desc: Any, parse_args: Optional[Dict] = None, construct_fn: str = 'build_from_arch_desc',
        fn_args: Optional[Dict] = None, substitute: bool = False, skip_exist: bool = True
    ) -> None:
        super().__init__(arch_desc, parse_args)
        self.construct_fn = construct_fn
        self.fn_args = fn_args or {}
        self.substitute = substitute
        self.skip_exist = skip_exist

    def visit(self, module: Module) -> Module:
        """Construct and return module."""
        construct_fn = getattr(module, self.construct_fn, None)
        if construct_fn is not None and not (isinstance(module, Slot) and module.get_entity() is not None):
            ret = construct_fn(self.arch_desc, **copy.deepcopy(self.fn_args))
            return module if ret is None else ret
        for n, m in module.named_children():
            m = self.visit(m)
            if m is not None and self.substitute:
                module.add_module(n, m)
        return module

    def __call__(self, model: Module) -> Module:
        """Run constructor."""
        Slot.set_convert_fn(self.convert)
        return self.visit(model)

    def convert(self, slot: Slot, desc: Sequence[str], *args, **kwargs) -> Module:
        """Convert Slot to module from archdesc."""
        if slot.get_entity() is not None and self.skip_exist:
            logger.warning('slot {} already built'.format(slot.sid))
            return None
        desc = desc[0] if isinstance(desc, list) else desc
        return build_module(desc, slot, *args, **kwargs)


@register
class DefaultSlotArchDescConstructor(DefaultSlotTraversalConstructor, DefaultArchDescConstructor):
    """Constructor that converts Slots to modules from archdesc."""

    def __init__(
        self, arch_desc: Any, parse_args: Optional[Dict] = None, construct_fn: str = 'build_from_arch_desc',
        fn_args: Optional[Dict] = None, traversal_args: Optional[Dict] = None, desc_args: Optional[Dict] = None
    ) -> None:
        DefaultSlotTraversalConstructor.__init__(self, **(traversal_args or {}))
        DefaultArchDescConstructor.__init__(self, arch_desc, parse_args, **(desc_args or {}))
        self.construct_fn = construct_fn
        self.fn_args = fn_args or {}
        self.idx = -1

    def get_next_desc(self) -> Any:
        """Return next archdesc item."""
        self.idx += 1
        desc = self.arch_desc[self.idx]
        if isinstance(desc, list) and len(desc) == 1:
            desc = desc[0]
        return desc

    def convert(self, slot: Slot, desc=None, *args, **kwargs) -> Module:
        """Convert Slot to module from archdesc."""
        if slot in self.visited:
            return None
        self.visited.add(slot)
        desc = desc or self.get_next_desc()
        ent = slot.get_entity()
        fn_args = copy.deepcopy(self.fn_args)
        if ent is not None:
            construct_fn = getattr(ent, self.construct_fn, None)
            if construct_fn is not None:
                ret = construct_fn(desc, **fn_args)
                return ent if ret is None else ret
        return build_module(desc, slot, *args, **fn_args, **kwargs)
