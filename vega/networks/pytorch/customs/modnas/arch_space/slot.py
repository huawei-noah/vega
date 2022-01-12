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

"""Slot module."""
from functools import partial
import copy
import torch.nn as nn
from modnas.registry.arch_space import register
from modnas.utils.logging import get_logger


logger = get_logger('arch_space')


def _simplify_list(data):
    return data[0] if isinstance(data, (list, tuple)) and len(data) == 1 else data


@register
class Slot(nn.Module):
    """Stub module that is converted to actual modules by Constructors."""

    _slots = []
    _slot_id = -1
    _convert_fn = None
    _export_fn = None

    def __init__(self,
                 *args,
                 _chn_in=None,
                 _chn_out=None,
                 _stride=None,
                 in_shape=None,
                 out_shape=None,
                 name=None,
                 **kwargs):
        super().__init__()
        Slot.register(self)
        self.name = str(self.sid) if name is None else name
        out_shape = out_shape or in_shape
        _chn_out = _chn_out or _chn_in
        if not isinstance(in_shape, (list, tuple)):
            in_shape = [in_shape]
        if not isinstance(out_shape, (list, tuple)):
            out_shape = [out_shape]
        if _chn_in is not None:
            _chn_in = [_chn_in] if not isinstance(_chn_in, (list, tuple)) else _chn_in
            in_shape = [([None, cin, None] if in_s is None else in_s) for in_s, cin in zip(in_shape, _chn_in)]
        if _chn_out is not None:
            _chn_out = [_chn_out] if not isinstance(_chn_out, (list, tuple)) else _chn_out
            out_shape = [([None, cout, None] if out_s is None else out_s) for out_s, cout in zip(out_shape, _chn_out)]
        strides = [([(o // i if i and o else None) for i, o in zip(in_s, out_s)] if in_s and out_s else None)
                   for in_s, out_s in zip(in_shape, out_shape)]
        if _stride is not None:
            strides = [([None, None, _stride] if strd is None else (strd[:1] + [(_stride if s is None else s)
                                                                                for s in strd[1:]]))
                       for strd in strides]
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.strides = strides
        self.ent = None
        self.kwargs = kwargs
        self.args = args
        logger.debug('slot {} {}: declared {} {} {}'.format(self.sid, self.name, self.in_shape, self.out_shape,
                                                            self.strides))

    @staticmethod
    def register(slot):
        """Register a Slot module."""
        slot.sid = Slot.new_slot_id()
        Slot._slots.append(slot)

    @staticmethod
    def reset():
        """Reset Slot modules."""
        Slot._slots = []
        Slot._slot_id = -1
        Slot._convert_fn = None
        Slot._export_fn = None

    @staticmethod
    def new_slot_id():
        """Return a new Slot id."""
        Slot._slot_id += 1
        return Slot._slot_id

    @property
    def chn_in(self):
        """Return input channel (first) dimension."""
        return _simplify_list([(None if in_s is None else in_s[1])
                               for in_s in self.in_shape]) if self.in_shape is not None else None

    @property
    def chn_out(self):
        """Return output channel (first) dimension."""
        return _simplify_list([(None if out_s is None else out_s[1])
                               for out_s in self.out_shape]) if self.out_shape is not None else None

    @property
    def stride(self):
        """Return strides of data dimensions."""
        return _simplify_list([(None if s is None else s[2])
                               for s in self.strides]) if self.strides is not None else None

    @staticmethod
    def gen_slots_all():
        """Return an iterator over all Slots."""
        for m in Slot._slots:
            yield m

    @staticmethod
    def gen_slots_model(model):
        """Return an iterator over all Slots in a model."""
        def gen():
            for m in model.modules():
                if isinstance(m, Slot):
                    yield m

        return gen

    @staticmethod
    def call_all(funcname, gen=None, fn_kwargs=None):
        """Call a member function over Slots."""
        if gen is None:
            gen = Slot.gen_slots_all
        ret = []
        for m in gen():
            if hasattr(m, funcname):
                ret.append(getattr(m, funcname)(**({} if fn_kwargs is None else copy.deepcopy(fn_kwargs))))
        return ret

    @staticmethod
    def apply_all(func, gen=None, fn_kwargs=None):
        """Apply a function over Slots."""
        if gen is None:
            gen = Slot.gen_slots_all
        ret = []
        for m in gen():
            ret.append(func(m, **({} if fn_kwargs is None else copy.deepcopy(fn_kwargs))))
        return ret

    @staticmethod
    def set_convert_fn(func):
        """Set default Slot convert function."""
        Slot._convert_fn = func

    @staticmethod
    def set_export_fn(func):
        """Set default Slot export function."""
        Slot._export_fn = func

    def get_entity(self):
        """Return actual module in Slot."""
        return self._modules.get('ent', None)

    def set_entity(self, ent):
        """Set actual module in Slot."""
        self.ent = ent
        logger.debug('slot {} {}: set to {}'.format(self.sid, self.name, ent.__class__.__name__))

    def del_entity(self):
        """Delete actual module in Slot."""
        if self.ent is None:
            return
        del self.ent

    def forward(self, *args, **kwargs):
        """Compute output."""
        if self.ent is None:
            raise RuntimeError('Undefined entity in slot {}'.format(self.sid))
        return self.ent(*args, **kwargs)

    def to_arch_desc(self, *args, **kwargs):
        """Return archdesc from Slot."""
        export_fn = Slot._export_fn
        if export_fn is None:
            logger.warning('slot {} has no exporter'.format(self.sid))
            return None
        return export_fn(self, *args, **kwargs)

    def build_from_arch_desc(self, *args, **kwargs):
        """Convert Slot to module from archdesc."""
        convert_fn = Slot._convert_fn
        if convert_fn is None:
            logger.warning('slot {} has no constructor'.format(self.sid))
            return
        ent = convert_fn(self, *args, **kwargs)
        if ent is not None:
            self.set_entity(ent)

    def extra_repr(self):
        """Return extra string representation."""
        return '{}, {}, {}, '.format(self.chn_in, self.chn_out, self.stride) + ', '.join(
            ['{}={}'.format(k, v) for k, v in self.kwargs.items()])


def get_slot_builder(builder, args_fmt=None, kwargs_fmt=None):
    """Return a builder that converts slot to module."""
    def get_slot_args(slot, args_fmt, kwargs_fmt):
        if args_fmt is None:
            return slot.args
        attr_map = {
            'i': 'in_shape',
            'o': 'out_shape',
            's': 'strides',
            'a': 'args',
        }
        attr_cnt = {k: 1 for k in attr_map.keys()}
        attr_cnt['s'] = 2
        args = []
        attr_type = None
        for i, c in enumerate(args_fmt):
            if attr_type is None:
                attr_type = c
            else:
                attr = getattr(slot, attr_map[attr_type])
                attr = _simplify_list(attr)
                last_attr, attr_type = attr_type, None
                if c in attr_map:
                    arg = attr[attr_cnt[last_attr]]
                    attr_cnt[last_attr] += 1
                    attr_type = c
                elif c.isdigit():
                    idx = int(c)
                    arg = attr[idx]
                elif c == '&':
                    arg = attr
                elif c == '*':
                    args.extend(attr)
                    continue
                else:
                    raise ValueError('invalid fmt')
                args.append(arg)
        kwargs = slot.kwargs if kwargs_fmt == '*' else {k: slot.kwargs[k] for k in (kwargs_fmt or [])}
        return args, kwargs

    def bld(slot, *args, **kwargs):
        s_args, s_kwargs = get_slot_args(slot, args_fmt, kwargs_fmt)
        return builder(*s_args, *args, **s_kwargs, **kwargs)

    return bld


def register_slot_builder(builder, _reg_id=None, args_fmt=None, kwargs_fmt=None):
    """Register a builder that converts slot to module."""
    _reg_id = _reg_id or builder.__qualname__
    register(get_slot_builder(builder, args_fmt, kwargs_fmt), _reg_id)
    return builder


register_slot_ccs = partial(register_slot_builder, args_fmt='i1o1s2')
