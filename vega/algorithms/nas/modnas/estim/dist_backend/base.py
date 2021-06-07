# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Distributed remote client and server."""
import threading


class RemoteBase():
    """Distributed remote client class."""

    def __init__(self):
        super().__init__()
        self.on_done = None
        self.on_failed = None

    def call(self, func, *args, on_done=None, on_failed=None, **kwargs):
        """Call function on remote client with callbacks."""
        self.on_done = on_done
        self.on_failed = on_failed
        self.th_rpc = threading.Thread(target=self.rpc, args=(func,) + args, kwargs=kwargs)
        self.th_rpc.start()

    def close(self):
        """Close the remote client."""
        raise NotImplementedError

    def rpc(self, func, *args, **kwargs):
        """Call function on remote client."""
        raise NotImplementedError

    def on_rpc_done(self, ret):
        """Invoke callback when remote call finishes."""
        self.ret = ret
        self.on_done(ret)

    def on_rpc_failed(self, ret):
        """Invoke callback when remote call fails."""
        self.on_failed(ret)


class WorkerBase():
    """Distributed remote worker (server) class."""

    def run(self, estim):
        """Run worker."""
        raise NotImplementedError

    def close(self):
        """Close worker."""
        raise NotImplementedError
