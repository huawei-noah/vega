# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Utils functions that been used in pipeline."""

import os
import socket
import logging
import signal
import psutil
from enum import Enum
from vega.common import FileOps
from vega.common.task_ops import TaskOps


class WorkerTypes(Enum):
    """WorkerTypes."""

    TRAINER = 1
    EVALUATOR = 2
    HOST_EVALUATOR = 3
    DeviceEvaluator = 5


# Here start the stand alone functions for master to use!
def clean_cuda_proc(master_pid, device_id):
    """Short summary.

    :param type master_pid: Description of parameter `master_pid`.
    :param type device_id: Description of parameter `device_id`.
    """
    current_pid = os.getpid()
    cuda_kill = "fuser -v /dev/nvidia{0} | " \
                "awk '{{for(i=1;i<=NF;i++)if($i!={1}&&$i!={2})" \
                "print \"kill -9 \" $i;}}' | sh".format(device_id, master_pid, current_pid)
    os.system(cuda_kill)
    return


def kill_proc_tree(pid, sig=signal.SIGKILL, include_parent=True,
                   timeout=None, on_terminate=None):
    """Kill a process tree (including grandchildren) with signal.

    "sig" and return a (gone, still_alive) tuple.
    "on_terminate", if specified, is a callabck function which is
    called as soon as a child terminates.
    """
    if pid == os.getpid():
        raise RuntimeError("I refuse to kill myself")
    gone = None
    alive = None
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        if include_parent:
            children.append(parent)
        for p in children:
            p.send_signal(sig)
        gone, alive = psutil.wait_procs(children, timeout=timeout,
                                        callback=on_terminate)
    except Exception:

        pass
    return (gone, alive)


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
        logging.info("get master address, address={}, ip={}, port={}".format(
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


def save_master_ip(ip_address, port, args):
    """Write the ip and port in a system path.

    :param str ip_address: The `ip_address` need to write.
    :param str port: The `port` need to write.
    :param argparse.ArgumentParser args: `args` is a argparse that should
         contain `init_method`, `rank` and `world_size`.

    """
    temp_folder = TaskOps().temp_path
    FileOps.make_dir(temp_folder)
    file_path = os.path.join(temp_folder, 'ip_address.txt')
    logging.info("write ip, file path={}".format(file_path))
    with open(file_path, 'w') as f:
        f.write(ip_address + "\n")
        f.write(port + "\n")


def load_master_ip():
    """Get the ip and port that write in a system path.

    here will not download anything from S3.
    """
    temp_folder = TaskOps().temp_path
    FileOps.make_dir(temp_folder)
    file_path = os.path.join(temp_folder, 'ip_address.txt')
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            ip = f.readline().strip()
            port = f.readline().strip()
            logging.info("get write ip, ip={}, port={}".format(
                ip, port
            ))
            return ip, port
    else:
        return None, None


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
