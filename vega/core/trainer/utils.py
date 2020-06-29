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
import subprocess
import sys
import logging
import signal
import psutil
from collections import OrderedDict
from enum import Enum
from vega.core.common.file_ops import FileOps
from vega.core.common.user_config import UserConfig


class WorkerTypes(Enum):
    """WorkerTypes."""

    TRAINER = 1
    EVALUATOR = 2
    GPU_EVALUATOR = 3
    HAVA_D_EVALUATOR = 4


class PairDictQueue():
    """A special Dict Queue only for Master to use to collect all finished Evaluator results.

    the insert and pop item could only be string or int.
    as a example for how to used in Evalutor, the stored odict could be :
    {
        "step_name::worker1": {"EVALUATE_GPU":0, "EVALUATE_DLOOP":0},
        "step_name::worker2": {"EVALUATE_GPU":0, "EVALUATE_DLOOP":1},
        "step_name::worker3": {"EVALUATE_GPU":1, "EVALUATE_DLOOP":0},
        "step_name::worker4": {"EVALUATE_GPU":1, "EVALUATE_DLOOP":1},
    }
    the list could mean each sub-evalutor-worker's status, 0 is not finished,
    1 is finished, here as example, this list could mean [gpu, dloop].
    and the key of odict is the id of this task(which combined with step name
    and worker-id).
    Only sub-evalutor-worker's all status turn to 1(finshed), could it be able
    to be popped from this PairDictQueue.

    :param int pair_size: Description of parameter `pair_size`.
    """

    def __init__(self):
        self.dq_id = 0
        self.odict = OrderedDict()
        return

    def add_new(self, item, type):
        """Short summary.

        :param type item: Description of parameter `item`.
        :param type key: Description of parameter `key`.
        """
        if item not in self.odict:
            self.odict[item] = dict()
        self.odict[item][type] = 0

    def put(self, item, type):
        """Short summary.

        :param type item: Description of parameter `item`.
        :param type type: Description of parameter `type`.
        :return: Description of returned object.
        :rtype: type

        """
        if item not in self.odict:
            logging.info("item({}) not in PairDictQueue!".format(item))
            return
        self.odict[item][type] = 1
        logging.info("PairDictQueue add item({}) key({})".format(item, type))
        return True

    def get(self):
        """Short summary.

        :return: Description of returned object.
        :rtype: type

        """
        item = None
        for key, subdict in self.odict.items():
            item_ok = True
            for k, i in subdict.items():
                if i != 1:
                    item_ok = False
                    break
            if item_ok:
                self.odict.pop(key)
                item = key
                break
        return item

    def qsize(self):
        """Short summary.

        :return: Description of returned object.
        :rtype: type

        """
        return len(self.odict)


# Here start the stand alone functions for master to use!
def clean_cuda_proc(master_pid, device_id):
    """Short summary.

    :param type master_pid: Description of parameter `master_pid`.
    :param type device_id: Description of parameter `device_id`.
    """
    current_pid = os.getpid()
    cuda_kill = "fuser -v /dev/nvidia{0} | "\
                "awk '{{for(i=1;i<=NF;i++)if($i!={1}&&$i!={2})"\
                "print \"kill -9 \" $i;}}' | sh".format(
                    device_id,
                    master_pid,
                    current_pid
                )
    os.system(cuda_kill)
    return


def kill_children_proc(sig=signal.SIGTERM, recursive=True,
                       timeout=1, on_terminate=None):
    """Kill a process tree of curret process (including grandchildren).

    with signal "sig" and return a (gone, still_alive) tuple.
    "on_terminate", if specified, is a callabck function which is
    called as soon as a child terminates.
    """
    pid = os.getpid()
    parent = psutil.Process(pid)
    children = parent.children(recursive)
    for p in children:
        logging.info("children: {}".format(p.as_dict(attrs=['pid', 'name', 'username'])))
        p.send_signal(sig)
    gone, alive = psutil.wait_procs(children, timeout=timeout,
                                    callback=on_terminate)
    return (gone, alive)


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


def install_and_import_local(package, package_path=None, update=False):
    """Install and import local python packages.

    :param str package: `package` name that need to install and import.
    :param package_path: if the package is a local whl, then the `package_path`.
    :type package_path: str or None
    :param bool update: Description of parameter `update`.

    """
    import importlib
    try:
        if not update:
            try:
                importlib.import_module(package)
            except ImportError:
                import pip
                if hasattr(pip, 'main'):
                    pip.main(['install', package_path])
                elif hasattr(pip, '_internal'):
                    pip._internal.main(['install', package_path])
                else:
                    subprocess.call([sys.executable, "-m", "pip", "install",
                                     package_path])
        else:
            import pip
            if hasattr(pip, 'main'):
                pip.main(['install', '-U', package_path])
            elif hasattr(pip, '_internal'):
                pip._internal.main(['install', '-U', package_path])
            else:
                subprocess.call([sys.executable, "-m", "pip", "install", "-U",
                                 package_path])
    finally:
        globals()[package] = importlib.import_module(package)


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


def write_ip(ip_address, port, args):
    """Write the ip and port in a system path.

    :param str ip_address: The `ip_address` need to write.
    :param str port: The `port` need to write.
    :param argparse.ArgumentParser args: `args` is a argparse that should
         contain `init_method`, `rank` and `world_size`.

    """
    local_base_path = UserConfig().data.general.task.local_base_path
    local_task_id = UserConfig().data.general.task.task_id
    local_path = os.path.join(local_base_path, local_task_id, 'ip_address/')
    if not os.path.exists(local_path):
        FileOps.make_dir(local_path)

    file_path = os.path.join(local_path, 'ip_address.txt')
    logging.info("write ip, file path={}".format(file_path))
    with open(file_path, 'w') as f:
        f.write(ip_address + "\n")
        f.write(port + "\n")


def get_write_ip(args):
    """Get the ip and port that write in a system path.

    :param argparse.ArgumentParser args: `args` is a argparse that should
         contain `init_method`, `rank` and `world_size`.
    :return: the ip and port .
    :rtype: str, str.

    """
    local_base_path = UserConfig().data.general.task.local_base_path
    local_task_id = UserConfig().data.general.task.task_id
    local_path = os.path.join(local_base_path, local_task_id, 'ip_address/')
    if not os.path.exists(local_path):
        FileOps.make_dir(local_path)
    file_path = os.path.join(local_path, 'ip_address.txt')
    with open(file_path, 'r') as f:
        ip = f.readline().strip()
        port = f.readline().strip()
        logging.info("get write ip, ip={}, port={}".format(
            ip, port
        ))
        return ip, port


def get_write_ip_master_local():
    """Get the ip and port that write in a system path.

    here will not download anything from S3.
    """
    local_base_path = UserConfig().data.general.task.local_base_path
    local_task_id = UserConfig().data.general.task.task_id
    local_path = os.path.join(local_base_path, local_task_id, 'ip_address/')
    file_path = os.path.join(local_path, 'ip_address.txt')
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
