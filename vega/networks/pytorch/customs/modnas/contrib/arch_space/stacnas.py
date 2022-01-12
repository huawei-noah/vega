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
