# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""StacNAS Constructors & Exporters."""
from modnas.registry.arch_space import build
from modnas.registry.construct import register
from modnas.registry.construct import DefaultMixedOpConstructor, DefaultRecursiveArchDescConstructor


@register
class StacNASArchDescSearchConstructor(DefaultRecursiveArchDescConstructor, DefaultMixedOpConstructor):
    """StacNAS Search Constructor."""

    def __init__(self, *args, arch_desc, candidates_map=None, **kwargs):
        DefaultRecursiveArchDescConstructor.__init__(self, arch_desc=arch_desc)
        DefaultMixedOpConstructor.__init__(self, *args, **kwargs)
        candidates_map = {
            'MAX': ['AVG', 'MAX', 'NIL'],
            'AVG': ['AVG', 'MAX', 'NIL'],
            'SC3': ['SC3', 'SC5', 'NIL'],
            'SC5': ['SC3', 'SC5', 'NIL'],
            'DC3': ['DC3', 'DC5', 'NIL'],
            'DC5': ['DC3', 'DC5', 'NIL'],
        } if candidates_map is None else candidates_map
        self.candidates_map = candidates_map

    def convert(self, slot, desc):
        """Convert Slot to mixed operator."""
        desc = desc[0] if isinstance(desc, list) else desc
        if desc in ['IDT', 'NIL']:
            ent = build(desc, slot)
        else:
            self.candidates = self.candidates_map[desc]
            ent = DefaultMixedOpConstructor.convert(self, slot)
        return ent
