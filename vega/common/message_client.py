# -*- coding: utf-8 -*-

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

"""Message Server."""

import logging
import zmq
from vega.common.json_coder import JsonEncoder
from vega.common.zmq_op import connect


__all__ = ["MessageClient"]
logger = logging.getLogger(__name__)


class MessageClient(object):
    """Message client."""

    def __init__(self, ip="127.0.0.1", port=None, timeout=30):
        """Initialize message client."""
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self._init_socket()

    def _init_socket(self):
        try:
            self.socket = connect(ip=self.ip, port=self.port)
            self.poller = zmq.Poller()
            self.poller.register(self.socket, zmq.POLLIN)
        except Exception as e:
            raise IOError(f"Failed to connect to tcp://{self.ip}:{self.port}, message: {e}")

    def _reset_socket(self):
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        self.poller.unregister(self.socket)
        self._init_socket()

    def send(self, action, data=None):
        """Send data."""
        try:
            if data:
                self.socket.send_json({"action": f"{action}", "data": f"{data}"}, cls=JsonEncoder)
            else:
                self.socket.send_json({"action": f"{action}"})

            socks = dict(self.poller.poll(self.timeout * 10000))
            if socks.get(self.socket) == zmq.POLLIN:
                msg = self.socket.recv_json()
                return msg
            else:
                self._reset_socket()
                raise IOError(f"Send message timeout, action: {action}, data: {data}")
        except Exception as e:
            raise IOError(f"Failed to send message, action: {action}, data: {data}, msg: {e}")
