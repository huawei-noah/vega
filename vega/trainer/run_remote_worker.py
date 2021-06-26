# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Run worker remotely."""

import os
import pickle
import psutil
import logging
import subprocess
import json
import traceback
import signal
import vega


def run_remote_worker(worker_id, worker_path, id, num_workers):
    """Run worker on remote node."""
    from vega.common.utils import init_log
    init_log(level="info",
             log_file=".temp_{}.log".format(worker_id),
             log_path=worker_path)
    for index in range(num_workers):
        config = _load_config(worker_id, worker_path, id, index)
        if "LD_LIBRARY_PATH" in config["env"] and config["env"]["LD_LIBRARY_PATH"] is not None:
            os.environ["LD_LIBRARY_PATH"] = config["env"]["LD_LIBRARY_PATH"]
        os.environ["PWD"] = config["env"]["PWD"]
        os.chdir(os.environ["PWD"])
        vega.set_backend(os.environ['BACKEND_TYPE'].lower(), os.environ["DEVICE_CATEGORY"])

        if vega.is_gpu_device():
            sub_pid_list = call_in_gpu(config, id, worker_id, worker_path, index)
        elif vega.is_npu_device():
            os.environ["PYTHONPATH"] = config["env"]["PYTHONPATH"]
            os.environ["PATH"] = config["env"]["PATH"]
            os.environ["ASCEND_OPP_PATH"] = config["env"]["ASCEND_OPP_PATH"]
            sub_pid_list = call_in_npu(config, id, worker_id, worker_path, index)
        logging.info("DistributedWorker finished!")
        for sub_pid in sub_pid_list:
            kill_proc_tree(pid=sub_pid)
        logging.info("DistributedWorker subprocess cleaned!")
    return 0


def _load_config(worker_id, worker_path, id, index):
    _config_file = os.path.join(
        worker_path,
        f".{str(id)}.{str(index)}.config.pkl")
    with open(_config_file, 'rb') as f:
        config = pickle.load(f)
    return config


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


def call_in_gpu(config, id, worker_id, worker_path, index):
    """Call function based on GPU devices."""
    env = os.environ.copy()
    sub_pid_list = []
    worker_nccl_port = config["worker_nccl_port"]
    world_size = config["world_size"]
    if 'CUDA_VISIBLE_DEVICES' in env:
        try:
            first_gpu_id = env['CUDA_VISIBLE_DEVICES'].split(",")[0]
            env['VEGA_WORKER_PORT'] = '{}'.format(worker_nccl_port + int(first_gpu_id))
        except Exception:
            env['VEGA_WORKER_PORT'] = '{}'.format(worker_nccl_port)
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = "{}:{}:{}".format(
            env['PYTHONPATH'], worker_path, os.path.abspath(os.curdir))
    elif worker_id is not None and worker_path is not None:
        env['PYTHONPATH'] = "{}:{}".format(
            worker_path, os.path.abspath(os.curdir))
    sub_pid = _subprocess(
        config, id, worker_id, worker_path, rank=0, world_size=world_size,
        env=env, is_backend=False, index=index)
    sub_pid_list.append(sub_pid)
    return sub_pid_list


def call_in_npu(config, id, worker_id, worker_path, index):
    """Call function based on NPU devices."""
    env = os.environ.copy()
    sub_pid_list = []
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = "{}:{}:{}".format(
            env['PYTHONPATH'], worker_path, os.path.abspath(os.curdir))
    elif worker_id is not None and worker_path is not None:
        env['PYTHONPATH'] = "{}:{}".format(
            worker_path, os.path.abspath(os.curdir))
    rank_file = env.get('RANK_TABLE_FILE')
    with open(rank_file, 'r') as f:
        rank_table_json = json.loads(f.read())
    if config["general"].get('dft', False):
        env['RANK_SIZE'] = env['ORIGIN_RANK_SIZE']
        env['RANK_TABLE_FILE'] = env['ORIGIN_RANK_TABLE_FILE']
    else:
        env['RANK_SIZE'] = '1'
        env['DEVICE_ID'] = rank_table_json['server_list'][0]['device'][0]['device_id']
        env['MASTER_ADDR'] = rank_table_json['server_list'][0]['device'][0]['device_ip']
        env['MASTER_PORT'] = rank_table_json['server_list'][0].get('server_port', '29688')
        env['RANK_ID'] = env['DEVICE_ID']
        env.pop('RANK_TABLE_FILE', None)
    from vega.common import switch_directory
    with switch_directory(worker_path):
        sub_pid = _subprocess(
            config, id, worker_id, worker_path, rank=0, world_size=1,
            env=env, is_backend=False, index=index)
    sub_pid_list.append(sub_pid)
    return sub_pid_list


def _subprocess(config, id, worker_id, worker_path, rank, world_size, env, is_backend, index):
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
    env['RANK'] = "{}".format(rank)
    env['WORLD_SIZE'] = "{}".format(world_size)

    _refresh_config_file(config, id, worker_id, worker_path, env, index)

    config_file = os.path.join(
        worker_path,
        f".{str(id)}.{str(index)}.config.pkl")
    worker_file = os.path.join(
        worker_path,
        f".{str(id)}.{str(index)}.worker.pkl")

    cmd = "from vega.trainer.deserialize import load_config;"
    cmd += "load_config('{}');".format(config_file)

    if 'VEGA_INIT_ENV' in os.environ:
        cmd += os.environ.copy()['VEGA_INIT_ENV']

    cmd += "from vega.trainer.deserialize import load_worker;"
    cmd += "worker=load_worker('{}');".format(worker_file)
    cmd += "worker.train_process();"

    python_command = config["general"].get('python_command')

    if is_backend:
        proc = subprocess.Popen([python_command, '-c', cmd], close_fds=True, env=env)
        pid = proc.pid
    else:
        try:
            proc = subprocess.Popen([python_command, '-c', cmd], env=env)
            pid = proc.pid
            proc.wait(timeout=config["timeout"])
        except Exception:
            logging.warn("Timeout worker has been killed.")
            logging.warn(traceback.print_exc())
    return pid


def _refresh_config_file(config, id, worker_id, worker_path, env, index):
    config["env"]["RANK"] = env.get("RANK", None)
    config["env"]["WORLD_SIZE"] = env.get("WORLD_SIZE", None)
    config["env"]["PYTHONPATH"] = env.get("PYTHONPATH", None)
    config["env"]["RANK_TABLE_FILE"] = env.get("RANK_TABLE_FILE", None)
    config["env"]["RANK_SIZE"] = env.get("RANK_SIZE", None)
    config["env"]["DEVICE_ID"] = env.get("DEVICE_ID", None)
    config["env"]["RANK_ID"] = env.get("RANK_ID", None)
    config["env"]["MASTER_ADDR"] = env.get("MASTER_ADDR", None)
    config["env"]["MASTER_PORT"] = env.get("MASTER_PORT", None)

    config_file = os.path.join(
        worker_path,
        f".{str(id)}.{str(index)}.config.pkl")
    with open(config_file, "wb") as f:
        pickle.dump(config, f)
