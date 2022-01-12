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

"""Managed list of remote clients."""
import threading
from modnas.registry.dist_remote import register as register_remote, build
from .base import RemoteBase


@register_remote
class ManagedRemotes(RemoteBase):
    """Managed remote client class."""

    def __init__(self, remote_conf):
        super().__init__()
        remotes = {}
        for k, conf in remote_conf.items():
            remotes[k] = build(conf)
        self.remotes = remotes
        self.idle = {k: True for k in remote_conf.keys()}
        self.idle_cond = threading.Lock()

    def add_remote(self, key, rmt):
        """Add remote to managed list."""
        self.remotes[key] = rmt
        self.idle[key] = True

    def del_remote(self, key):
        """Remove remote from managed list."""
        del self.remotes[key]
        del self.idle[key]

    def is_idle(self, key):
        """Return if the remote is idle."""
        return self.idle[key]

    def idle_remotes(self):
        """Return list of idle remotes."""
        return [k for k, v in self.idle.items() if v]

    def get_idle_remote(self, busy=True, wait=True):
        """Return an idle remote."""
        idle_rmt = None
        while idle_rmt is None:
            idles = self.idle_remotes()
            if not idles:
                if not wait:
                    return None
                self.idle_cond.acquire()
                self.idle_cond.release()
            else:
                idle_rmt = idles[0]
        if busy:
            self.set_idle(idle_rmt, False)
        return idle_rmt

    def set_idle(self, key, idle=True):
        """Set remote idle state."""
        self.idle[key] = idle
        if idle and self.idle_cond.locked():
            self.idle_cond.release()
        elif not self.idle_remotes():
            self.idle_cond.acquire()

    def close(self):
        """Close the remote client."""
        for rmt in self.remotes.values():
            rmt.close()

    def call(self, *args, on_done=None, on_failed=None, **kwargs):
        """Call function on remote client with callbacks."""
        def wrap_cb(cb, r):
            def wrapped(*args, **kwargs):
                self.set_idle(r)
                return None if cb is None else cb(*args, **kwargs)
            return wrapped

        rmt_key = self.get_idle_remote()
        if rmt_key is None:
            return
        on_done = wrap_cb(on_done, rmt_key)
        on_failed = wrap_cb(on_failed, rmt_key)
        self.remotes[rmt_key].call(*args, on_done=on_done, on_failed=on_failed, **kwargs)
