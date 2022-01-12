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
import ast
import os
from threading import Thread
from vega.common.utils import singleton
from vega.common import JsonEncoder
from vega.common.zmq_op import listen


__all__ = ["MessageServer"]
logger = logging.getLogger(__name__)


@singleton
class MessageServer(object):
    """Message server."""

    def __init__(self):
        """Initialize message server."""
        self.handlers = {}
        self.min_port = 27000
        self.max_port = 27999
        self.port = None
        self.register_handler("query_task_info", query_task_info)

    def run(self, ip="*"):
        """Run message server."""
        if self.port is not None:
            return

        try:
            (socket, self.port) = listen(
                ip=ip, min_port=self.min_port, max_port=self.max_port, max_tries=100)
            logging.debug("Start message monitor thread.")
            monitor_thread = Thread(target=_monitor_socket, args=(socket, self.handlers))
            monitor_thread.daemon = True
            monitor_thread.start()
            return self.port
        except Exception as e:
            logging.error("Failed to run message monitor thread.")
            raise e

    def register_handler(self, action, function):
        """Register messge handler."""
        self.handlers[action] = function


def _monitor_socket(socket, handlers):
    while True:
        message = socket.recv_json()
        logger.debug("Message arrived: {message}")
        if "action" not in message:
            socket.send_json({"result": "failed", "message": "Invalid request."}, cls=JsonEncoder)
            continue

        action = message.get("action")
        if action not in handlers:
            socket.send_json({"result": "failed", "message": f"Invalid action {action}."}, cls=JsonEncoder)
            continue

        data = message.get("data", None)
        if isinstance(data, str):
            try:
                data = ast.literal_eval(data)
            except Exception as e:
                socket.send_json({"result": "failed", "message": f"{e}"}, cls=JsonEncoder)
                continue

        try:
            if isinstance(data, dict):
                result = handlers[action](**data)
            elif "data" in message:
                result = handlers[action](data)
            else:
                result = handlers[action]()
            socket.send_json(result, cls=JsonEncoder)
        except Exception as e:
            socket.send_json({"result": "failed", "message": f"{e}"}, cls=JsonEncoder)


def query_task_info():
    """Get task message."""
    from vega.common import TaskOps
    return {
        "result": "success",
        "task_id": TaskOps().task_id,
        "base_path": os.path.abspath(TaskOps().task_cfg.local_base_path),
    }
