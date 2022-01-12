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

"""Run worker remotely."""

import os
import sys
import logging
import traceback
import subprocess
import signal
import psutil
import vega
from vega.common.general import General
from vega.trainer.deserialize import load_config, load_worker
from vega import security


def run_remote_worker(worker_id, worker_path, id, num_workers):
    """Run worker on remote node."""
    from vega.common.utils import init_log, close_log
    fh = init_log(level="info",
                  log_file=".temp_{}.log".format(worker_id),
                  log_path=worker_path)
    for index in range(num_workers):
        os.chdir(os.environ["PWD"])
        if "PYTHONPATH" not in os.environ:
            os.environ["PYTHONPATH"] = ""
        if worker_path is not None and worker_path not in os.environ["PYTHONPATH"].split(":"):
                os.environ["PYTHONPATH"] += f":{worker_path}"
        if os.path.abspath(os.curdir) not in os.environ["PYTHONPATH"].split(":"):
            os.environ["PYTHONPATH"] += f":{os.path.abspath(os.curdir)}"

        if vega.is_gpu_device():
            sub_pid_list = call_in_gpu(id, worker_id, worker_path, index)
        elif vega.is_npu_device():
            sub_pid_list = call_in_npu(id, worker_id, worker_path, index)
        logging.info("DistributedWorker finished!")
        for sub_pid in sub_pid_list:
            kill_proc_tree(pid=sub_pid)
        logging.info("DistributedWorker subprocess cleaned!")
    close_log(fh)
    return 0


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
        logging.debug('Failed to a process tree.')
    return (gone, alive)


def call_in_gpu(id, worker_id, worker_path, index):
    """Call function based on GPU devices."""
    sub_pid_list = []
    sub_pid = _subprocess(
        id, worker_id, worker_path, rank=0, is_backend=False, index=index)
    sub_pid_list.append(sub_pid)
    return sub_pid_list


def call_in_npu(id, worker_id, worker_path, index):
    """Call function based on NPU devices."""
    sub_pid_list = []
    from vega.common import switch_directory
    with switch_directory(worker_path):
        sub_pid = _subprocess(
            id, worker_id, worker_path, rank=0, is_backend=False, index=index)
    sub_pid_list.append(sub_pid)
    return sub_pid_list


def _subprocess(id, worker_id, worker_path, rank, is_backend, index):
    """Subprocess on each rank.

    Load pickle file into worker class, and use subprocess to run the
    train_process function.

    :param rank: node rank
    :type rank: int
    :param world_size: number of total nodes
    :type world_size: int
    :param env: environ
    :type env: dict
    :param is_backend: backend or not
    :type is_backend: bool
    """
    config_file = os.path.join(
        worker_path,
        f".{str(id)}.{str(index)}.config.pkl")
    worker_file = os.path.join(
        worker_path,
        f".{str(id)}.{str(index)}.worker.pkl")

    python_command = os.environ["vega_python_command"]
    if is_backend:
        proc = subprocess.Popen(
            [python_command, '-m', "vega.trainer.run_remote_worker", config_file, worker_file],
            close_fds=True,
            env=os.environ.copy())
        pid = proc.pid
    else:
        try:
            proc = subprocess.Popen(
                [python_command, '-m', "vega.trainer.run_remote_worker", config_file, worker_file],
                env=os.environ.copy())
            pid = proc.pid
            proc.wait(timeout=int(os.environ["vega_timeout"]))
        except Exception as e:
            logging.warn(f"Timeout worker has been killed, message: {e}.")
            logging.debug(traceback.print_exc())
    return pid


def run_worker():
    """Run worker."""
    try:
        vega.set_backend(os.environ["BACKEND_TYPE"].lower(), os.environ["DEVICE_CATEGORY"])
        (config_file, worker_file) = sys.argv[1:]
        load_config(config_file)
        if General.security:
            if not security.load_config("client"):
                return
            os.umask(0o077)
        worker = load_worker(worker_file)
        worker.train_process()
    except Exception as e:
        logging.debug(traceback.format_exc())
        logging.error(f"Failed to run worker, message: {e}")


if __name__ == "__main__":
    run_worker()
