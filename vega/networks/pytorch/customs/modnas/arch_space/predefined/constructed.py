# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
