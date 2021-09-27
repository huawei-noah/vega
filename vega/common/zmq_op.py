# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ZMQ operation."""

import zmq


def listen(ip, min_port, max_port, max_tries):
    """Listen on the server."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port(
        f"tcp://{ip}", min_port=min_port, max_port=max_port, max_tries=100)
    return socket, port


def connect(ip, port):
    """Connect to server."""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{ip}:{port}")
    return socket
