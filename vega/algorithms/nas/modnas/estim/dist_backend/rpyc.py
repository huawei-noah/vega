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

"""RPyC remote server and client."""
import rpyc
from rpyc.utils.server import ThreadedServer
from modnas.registry.dist_remote import register as register_remote
from modnas.registry.dist_worker import register as register_worker
from .base import RemoteBase


@register_remote
class RPyCRemote(RemoteBase):
    """RPyC remote client class."""

    def __init__(self, address, port=18861):
        super().__init__()
        self.conn = rpyc.connect(address, port)

    def close(self):
        """Close the remote client."""
        try:
            self.conn.root.close()
        except EOFError:
            pass
        self.conn.close()

    def rpc(self, func, *args, **kwargs):
        """Call function on remote client."""
        ret = self.conn.root.estim_call(func, *args, **kwargs)
        self.on_rpc_done(ret)


def _convert_normal(obj):
    if isinstance(obj, dict):
        return {k: obj[k] for k in obj}
    return obj


class ModNASService(rpyc.Service):
    """RPyC remote service for modnas."""

    def exposed_get_estim(self):
        """Return estimator."""
        return self.estim

    def exposed_close(self):
        """Close server."""
        self.server.close()

    def exposed_estim_call(self, func, *args, **kwargs):
        """Return result of estimator call."""
        args = [_convert_normal(a) for a in args]
        kwargs = {k: _convert_normal(v) for k, v in kwargs.items()}
        return getattr(self.estim, func)(*args, **kwargs)


@register_worker
class RPyCWorker():
    """RPyC worker class."""

    def __init__(self, *args, port=18861, **kwargs):
        self.server = ThreadedServer(ModNASService, *args, port=port, **kwargs)

    def run(self, estim):
        """Run worker."""
        self.server.service.estim = estim
        self.server.service.server = self.server
        self.server.start()

    def close(self):
        """Close worker."""
        self.server.close()
