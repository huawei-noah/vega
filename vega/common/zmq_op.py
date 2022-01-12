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

"""ZMQ operation."""

import zmq
from vega.common import General
from vega.common.task_ops import TaskOps


def listen(ip, min_port, max_port, max_tries):
    """Listen on the server."""
    if not General.security:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        port = socket.bind_to_random_port(
            f"tcp://{ip}", min_port=min_port, max_port=max_port, max_tries=100)
        return socket, port
    else:
        from vega.security.zmq_op import listen_security
        temp_path = TaskOps().temp_path
        return listen_security(ip, min_port, max_port, max_tries, temp_path)


def connect(ip, port):
    """Connect to server."""
    if not General.security:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{ip}:{port}")
        return socket
    else:
        from vega.security.zmq_op import connect_security
        temp_path = TaskOps().temp_path
        return connect_security(ip, port, temp_path)
