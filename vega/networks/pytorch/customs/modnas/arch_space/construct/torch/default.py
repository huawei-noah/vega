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

"""Default Constructors."""
import importlib
import copy
from functools import partial
from collections import OrderedDict
from modnas.registry.arch_space import build as build_module
from modnas.registry.construct import register, build
from modnas.arch_space.slot import Slot
from modnas.utils.logging import get_logger
from modnas.utils import import_file


logger = get_logger('construct')


def get_convert_fn(convert_fn, **kwargs):
    """Return a new convert function."""
    if isinstance(convert_fn, str):
        return build(convert_fn, **kwargs)
    elif callable(convert_fn):
        return convert_fn
    else:
        raise ValueError('unsupported convert_fn type: {}'.format(type(convert_fn)))


@register
class DefaultModelConstructor():
    """Constructor that builds model from registered architectures."""

    def __init__(self, model_type, args=None):
        self.model_type = model_type
        self.args = args or {}

    def __call__(self, model):
        """Run constructor."""
        model = build_module(self.model_type, **copy.deepcopy(self.args))
        return model


@register
class ExternalModelConstructor():
    """Constructor that builds model from external sources or libraries."""

    def __init__(self, model_type, src_path=None, import_path=None, args=None):
        self.model_type = model_type
        self.import_path = import_path
        self.src_path = src_path
        self.args = args or {}

    def __call__(self, model):
        """Run constructor."""
        if self.src_path is not None:
            logger.info('Importing model from path: {}'.format(self.src_path))
            mod = import_file(self.src_path)
        elif self.import_path is not None:
            logger.info('Importing model from lib: {}'.format(self.import_path))
            mod = importlib.import_module(self.import_path)
        model = mod.__dict__[self.model_type](**self.args)
        return model


@register
class DefaultTraversalConstructor():
    """Constructor that traverses and converts modules."""

    def __init__(self, by_class=None, by_classname=None):
        self.by_class = by_class
        self.by_classname = by_classname

    def convert(self, module):
        """Return converted module."""
        raise NotImplementedError

    def __call__(self, model):
        """Run constructor."""
        for m in model.modules():
            for k, sm in m._modules.items():
                if self.by_class and not isinstance(sm, self.by_class):
                    continue
                if self.by_classname and type(sm).__qualname__ != self.by_classname:
                    continue
                new_sm = self.convert(sm)
                if new_sm is not None:
                    m._modules[k] = new_sm
        return model


@register
class DefaultSlotTraversalConstructor():
    """Constructor that traverses and converts Slots."""

    def __init__(self, gen=None, convert_fn=None, args=None, skip_exist=True):
        self.gen = gen
        self.skip_exist = skip_exist
        if convert_fn:
            self.convert = get_convert_fn(convert_fn, **(args or {}))
        self.visited = set()

    def convert(self, slot):
        """Return converted module from slot."""
        raise NotImplementedError

    def __call__(self, model):
        """Run constructor."""
        Slot.set_convert_fn(self.convert)
        gen = self.gen or Slot.gen_slots_model(model)
        all_slots = list(gen())
        for m in all_slots:
            if self.skip_exist and m.get_entity() is not None:
                continue
            ent = self.convert(m)
            if ent is not None:
                m.set_entity(ent)
        self.visited.clear()
        return model


@register
class DefaultMixedOpConstructor(DefaultSlotTraversalConstructor):
    """Default Mixed Operator Constructor."""

    def __init__(self, candidates, mixed_op, candidate_args=None):
        DefaultSlotTraversalConstructor.__init__(self)
        self.candidates = candidates
        self.mixed_op_conf = mixed_op
        self.candidate_args = candidate_args or {}

    def convert(self, slot):
        """Return converted MixedOp from slot."""
        cand_args = self.candidate_args.copy()
        candidates = self.candidates
        if isinstance(candidates, (list, tuple)):
            candidates = {k: k for k in candidates}
        cands = OrderedDict([(k, build_module(v, slot=slot, **cand_args)) for k, v in candidates.items()])
        return build_module(self.mixed_op_conf, candidates=cands)


@register
class DefaultOpConstructor(DefaultSlotTraversalConstructor):
    """Default Network Operator Constructor."""

    def __init__(self, op):
        DefaultSlotTraversalConstructor.__init__(self)
        self.op_conf = op

    def convert(self, slot):
        """Return converted operator from slot."""
        return build_module(copy.deepcopy(self.op_conf), slot)


def make_traversal_constructor(convert_fn):
    """Return default slot traversal constructor with given function as converter."""
    return partial(DefaultSlotTraversalConstructor, convert_fn=convert_fn)
