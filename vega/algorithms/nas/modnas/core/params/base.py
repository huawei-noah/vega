# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Base parameter."""
from collections import OrderedDict
from modnas.core.event import event_emit, event_on
from modnas.core.param_space import ParamSpace


class Param():
    """Base parameter class."""

    def __init__(self, name=None, space=None, on_update=None):
        self.name = None
        self._parent = None
        self._children = OrderedDict()
        space = space or ParamSpace()
        space.register(self, name)
        self.event_name = 'update:{}'.format(self.name)
        if on_update is not None:
            event_on(self.event_name, on_update)
        set_value_ori = self.set_value

        def set_value_hooked(*args, **kwargs):
            set_value_ori(*args, **kwargs)
            self.on_update()
        self.set_value = set_value_hooked

    def __repr__(self):
        """Return representation string."""
        return '{}(name={}, {})'.format(self.__class__.__name__, self.name, self.extra_repr())

    def extra_repr(self):
        """Return extra representation string."""
        return ''

    def is_valid(self, value):
        """Return if the value is valid."""
        return True

    def value(self):
        """Return parameter value."""
        return self.val

    def set_value(self, value):
        """Set parameter value."""
        if not self.is_valid(value):
            raise ValueError('Invalid parameter value')
        self.val = value

    def on_update(self):
        """Trigger parameter update event."""
        event_emit(self.event_name, self)

    def __deepcopy__(self, memo):
        """Return deepcopy."""
        # disable deepcopy
        return self
