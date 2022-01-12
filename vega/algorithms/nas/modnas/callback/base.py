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

"""Base callback."""
from typing import Callable, Dict, Optional, Tuple, Union
from modnas.core.event import event_on, event_off
from modnas.utils.logging import get_logger


_HANDLER_CONF_TYPE = Dict[str, Union[Tuple[Callable, int], Callable]]


class CallbackBase():
    """Base callback class."""

    logger = get_logger('callback')
    priority = 0

    def __init__(self, handler_conf: Optional[_HANDLER_CONF_TYPE] = None) -> None:
        self.handlers = {}
        if handler_conf is not None:
            self.bind_handlers(handler_conf)

    def bind_handlers(self, handler_conf: _HANDLER_CONF_TYPE) -> None:
        """Bind event handlers."""
        handlers = {}
        for ev, conf in handler_conf.items():
            prio = None
            if isinstance(conf, (list, tuple)):
                h = conf[0]
                if len(conf) > 1:
                    prio = conf[1]
            else:
                h = conf
            event_on(ev, h, self.priority if prio is None else prio)
            handlers[ev] = h
        self.handlers = handlers

    def unbind_handlers(self):
        """Un-bind event handlers."""
        for ev, h in self.handlers.items():
            event_off(ev, h)
