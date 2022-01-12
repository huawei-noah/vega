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

"""Utils functions that been used in pipeline."""

import socket
import logging
from enum import Enum


class WorkerTypes(Enum):
    """WorkerTypes."""

    TRAINER = 1
    EVALUATOR = 2
    HOST_EVALUATOR = 3
    DeviceEvaluator = 5


def get_master_address(args):
    """Get master address(ip, port) from `args.init_method`.

    :param argparse.ArgumentParser args: `args` is a argparse that should
         contain `init_method`, `rank` and `world_size`.
    :return: ip, port.
    :rtype: (str, str) or None

    """
    if args.init_method is not None:
        address = args.init_method[6:].split(":")
        ip = socket.gethostbyname(address[0])
        port = address[-1]
        logging.debug("get master address, address={}, ip={}, port={}".format(
            address, ip, port
        ))
        return ip, port
    else:
        logging.warn("fail to get master address, args.init_method is none.")
        return None


def get_local_address():
    """Try to get the local node's IP.

    :return str: ip address.

    """
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    logging.info("get local address, hostname={}, ip={}".format(
        hostname, ip
    ))
    return ip


def get_master_port(args):
    """Get master port from `args.init_method`.

    :param argparse.ArgumentParser args: `args` is a argparse that should
         contain `init_method`, `rank` and `world_size`.
    :return: The port that master used to communicate with slaves.
    :rtype: str or None

    """
    if args.init_method is not None:
        address = args.init_method.split(":")
        port = address[-1]
        return port
    else:
        return None
