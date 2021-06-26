# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Message Server."""

import logging
import zmq
import ast
import os
from threading import Thread
from vega.common.utils import singleton
from vega.common import JsonEncoder


__all__ = ["MessageServer"]
logger = logging.getLogger(__name__)


@singleton
class MessageServer(object):
    """Message server."""

    def __init__(self):
        """Initialize message server."""
        self.handlers = {}
        self.min_port = 5000
        self.max_port = 7000
        self.port = None
        self.register_handler("query_task_info", query_task_info)

    def run(self, ip="*"):
        """Run message server."""
        if self.port is not None:
            return

        try:
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            self.port = socket.bind_to_random_port(
                f"tcp://{ip}", min_port=self.min_port, max_port=self.max_port, max_tries=100)
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
    from vega.common.task_ops import TaskOps
    return {
        "result": "success",
        "task_id": TaskOps().task_id,
        "base_path": os.path.abspath(TaskOps().task_cfg.local_base_path),
    }
