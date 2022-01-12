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

"""Base parameter."""
from collections import OrderedDict
from typing import Any, Dict, Optional, Union, Callable
from modnas.core.event import event_emit, event_on
from modnas.core.param_space import ParamSpace


class Param():
    """Base parameter class."""

    def __init__(
        self, name: Optional[str] = None, space: Optional[ParamSpace] = None, on_update: Optional[Callable] = None
    ) -> None:
        self.name = None
        self._parent = None
        self._children = OrderedDict()
        (space or ParamSpace()).register(self, name)
        self.event_name = 'update:{}'.format(self.name)
        if on_update is not None:
            event_on(self.event_name, on_update)
        set_value_ori = self.set_value

        def set_value_hooked(*args, **kwargs):
            set_value_ori(*args, **kwargs)
            self.on_update()
        self.set_value = set_value_hooked

    def __repr__(self) -> str:
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

    def on_update(self) -> None:
        """Trigger parameter update event."""
        event_emit(self.event_name, self)

    def __deepcopy__(self, memo: Dict[Union[int, str], Any]) -> Any:
        """Return deepcopy."""
        # disable deepcopy
        return self
