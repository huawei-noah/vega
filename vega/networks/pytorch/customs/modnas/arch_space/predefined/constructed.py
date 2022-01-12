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

"""Constructed modules."""
from modnas.registry.construct import build as build_constructor
from modnas.registry.arch_space import build as build_module
from modnas.registry.arch_space import register
from modnas.registry import streamline_spec


@register
def Constructed(slot=None, construct=None, module=None):
    """Return a module from constructors."""
    m = None if module is None else build_module(module, slot=slot)
    for con in streamline_spec(construct):
        m = build_constructor(con)(m)
    return m
