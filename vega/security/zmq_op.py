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
import os
import uuid
import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator


def listen_security(ip, min_port, max_port, max_tries, temp_path):
    """Listen on server."""
    ctx = zmq.Context.instance()
    # Start an authenticator for this context.
    auth = ThreadAuthenticator(ctx)
    auth.start()
    auth.configure_curve(domain='*', location=zmq.auth.CURVE_ALLOW_ANY)

    socket = ctx.socket(zmq.REP)
    server_secret_key = os.path.join(temp_path, "server.key_secret")
    if not os.path.exists(server_secret_key):
        _, server_secret_key = zmq.auth.create_certificates(temp_path, "server")
    server_public, server_secret = zmq.auth.load_certificate(server_secret_key)
    if os.path.exists(server_secret_key):
        os.remove(server_secret_key)
    socket.curve_secretkey = server_secret
    socket.curve_publickey = server_public
    socket.curve_server = True  # must come before bind

    port = socket.bind_to_random_port(
        f"tcp://{ip}", min_port=min_port, max_port=max_port, max_tries=100)
    return socket, port


def connect_security(ip, port, temp_path):
    """Connect to server."""
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.REQ)
    client_name = uuid.uuid1().hex[:8]
    client_secret_key = os.path.join(temp_path, "{}.key_secret".format(client_name))
    if not os.path.exists(client_secret_key):
        client_public_key, client_secret_key = zmq.auth.create_certificates(temp_path, client_name)
    client_public, client_secret = zmq.auth.load_certificate(client_secret_key)
    socket.curve_secretkey = client_secret
    socket.curve_publickey = client_public
    server_public_key = os.path.join(temp_path, "server.key")
    if not os.path.exists(server_public_key):
        server_public_key, _ = zmq.auth.create_certificates(temp_path, "server")
    server_public, _ = zmq.auth.load_certificate(server_public_key)
    socket.curve_serverkey = server_public
    socket.connect(f"tcp://{ip}:{port}")
    if os.path.exists(client_secret_key):
        os.remove(client_secret_key)
    if os.path.exists(client_public_key):
        os.remove(client_public_key)
    return socket
