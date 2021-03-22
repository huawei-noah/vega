# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Verify cluster env."""

import os
import argparse
import subprocess
import uuid
import time
import psutil
import shutil
import signal
from dask.distributed import Client


def _parse_args():
    desc = "Verify cluster."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-m", "--master", default=None, type=str, required=True,
                        help="Master node IP.")
    parser.add_argument("-s", "--slaves", dest="slaves", nargs="+", required=True,
                        help="Slaves node IP, eg. -s 192.168.0.2 192.168.0.3")
    parser.add_argument("-p", "--port", default=8786, type=int,
                        help="Listening port, default is 8786.")
    parser.add_argument("-n", "--nfs_folder", default=None, type=str, required=True,
                        help="Shared NFS folder.")
    args = parser.parse_args()
    return args


def _verify_ip(args):
    print("*" * 32)
    print("Start verify IP.")
    for slave in args.slaves:
        result = subprocess.call(["ping", "-c", "4", "{}".format(slave)])
        if result != 0:
            raise ValueError("Failed to access slave ({}).".format(slave))
        cmd = "ssh -o NumberOfPasswordPrompts=0 -o StrictHostKeyChecking=yes {} \"echo OK\"".format(slave)
        result = subprocess.call(cmd, shell=True)
        if result != 0:
            raise ValueError("Failed to login slave ({}) without password.".format(slave))
    print("Pass.")


def _verify_nfs(args):
    print("*" * 32)
    print("Start verify NFS.")
    if not os.path.exists(args.nfs_folder):
        raise ValueError("Shared NFS folder({}) is not existed.".format(args.nfs_folder))
    for slave in args.slaves:
        temp_folder = os.path.join(args.nfs_folder, uuid.uuid1().hex)
        cmd = "mkdir {}".format(temp_folder)
        result = subprocess.call(["ssh", slave, cmd])
        if result != 0:
            raise ValueError("Shared NFS folder ({}:{}) is not accessed.".format(slave, args.nfs_folder))
        cmd = "rm -r {}".format(temp_folder)
        result = subprocess.call(["ssh", slave, cmd])
        if result != 0:
            raise ValueError("Shared NFS folder ({}:{}) is not accessed.".format(slave, args.nfs_folder))
    print("Pass.")


def _kill_existed_dask(args):
    pids = psutil.pids()
    dask_pids = []
    for pid in pids:
        try:
            process = psutil.Process(pid)
            pname = process.name()
            if "dask-scheduler" in pname or "dask-worker" in pname:
                dask_pids.append(pid)
        except Exception:
            pass
    if dask_pids:
        print("Found existed dask scheduler or dask worker processes.")
        _input = input("Do you want kill dask processes and continue to verify? [Y/n]: ")
        if _input.upper() in ["N", "NO"]:
            print("Cluster verification canceled.")
            os._exit(0)
        elif _input.upper() not in ["", "Y", "YES"]:
            print("Input Error.")
            os._exit(0)
        for pid in dask_pids:
            os.kill(int(pid), signal.SIGKILL)
        time.sleep(10)


def _init_dask_scheduler(args):
    print("Start verify scheduler.")
    result = subprocess.Popen(["dask-scheduler", "--port", str(args.port)], close_fds=True)
    if not isinstance(result, subprocess.Popen):
        print("Failed to start dask scheduler.")
        print("Please run the command in CLI, and resovlue the problems.")
        print("dask-scheduler --port {}".format(args.port))
        raise ValueError("Failed to start dask scheduler.")
    time.sleep(5)
    print("Pass.")


def _verfiy_local(args):
    print("Start verify local worker, IP:{}, port: {}.".format(args.master, args.port))
    result = subprocess.Popen(["dask-worker", "{}:{}".format(args.master, args.port)], close_fds=True)
    if not isinstance(result, subprocess.Popen):
        raise ValueError("Can not start local dask-worker.")
    time.sleep(5)
    print("Pass.")
    print("Test local dask Client.")
    cmd = "python3 -c \"from dask.distributed import Client;client=Client('{}:{}');client.close()\"".format(
        args.master, args.port)
    result = subprocess.call(cmd, shell=True)
    if result != 0:
        raise ValueError("Can not start local dask client.")
    print("Pass.")


def _verify_client(args):
    print("Start verify slave workers.")
    for slave in args.slaves:
        print("Start verify slave worker, IP: {}.".format(slave))
        cmd = "{} {}:{}".format(shutil.which("dask-worker"), args.master, args.port)
        result = subprocess.Popen(["ssh", slave, cmd], close_fds=True, env=os.environ)
        if not isinstance(result, subprocess.Popen):
            raise ValueError("Can not start local dask-worker.")
        time.sleep(5)
        print("Pass.")
        print("Test slave dask Client, IP: {}.".format(slave))
        cmd = "python3 -c \"from dask.distributed import Client;client=Client('{}:{}');client.close()\"".format(
            args.master, args.port)
        result = subprocess.call(cmd, shell=True, env=os.environ)
        if result != 0:
            raise ValueError("Can not start local dask client.")
        time.sleep(5)
        print("Pass.")
    print("Pass.")


def _stop_dask_scheduler(args):
    print("Start stop scheduler.")
    client = Client("{}:{}".format(args.master, args.port))
    try:
        client.shutdown()
        client.close()
        del client
        # print("Scheduler stopped.")
        time.sleep(8)
    except Exception:
        print("Failed to stop scheduler, please stop it manually.")


def _verify_dask(args):
    print("*" * 32)
    # print("Start verify Dask worker and scheduler.")
    _kill_existed_dask(args)
    _init_dask_scheduler(args)
    _verfiy_local(args)
    _verify_client(args)
    _stop_dask_scheduler(args)
    print("Pass.")


def _verify_cluster():
    args = _parse_args()
    print(args)
    _verify_ip(args)
    _verify_nfs(args)
    _verify_dask(args)
    print("All cluster check items have passed.")


if __name__ == "__main__":
    _verify_cluster()
