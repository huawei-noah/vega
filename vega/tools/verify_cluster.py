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
import subprocess
import uuid
import time
import psutil
import shutil
import signal
import json
from dask.distributed import Client
from vega.common import argment_parser
from vega.common.general import General
from vega.common.utils import get_available_port


def _parse_args():
    parser = argment_parser("Verify cluster.")
    parser.add_argument("-m", "--master", default=None, type=str, required=True,
                        help="master node IP")
    parser.add_argument("-s", "--slaves", dest="slaves", nargs="+", required=True,
                        help="slaves node IP, eg. -s 192.168.0.2 192.168.0.3")
    parser.add_argument("-n", "--nfs_folder", default=None, type=str, required=True,
                        help="shared NFS folder")
    parser.add_argument("-j", "--json", action='store_true',
                        help="silence mode, print result with json format")
    args = parser.parse_args()
    return args


_json = None
_port = None


def _print(value):
    global _json
    if not _json:
        print(value)


def _call(cmd, **kwargs):
    global _json
    if _json:
        return subprocess.call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    else:
        return subprocess.call(cmd, **kwargs)


def _check_output(cmd):
    global _json
    if _json:
        return subprocess.check_output(cmd, stderr=subprocess.PIPE).decode("utf-8")
    else:
        return subprocess.check_output(cmd).decode("utf-8")


def _popen(cmd):
    global _json
    if _json:
        return subprocess.Popen(cmd, close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        return subprocess.Popen(cmd, close_fds=True)


def _verify_ip(args):
    _print("*" * 32)
    _print("Start verify IP.")
    for slave in args.slaves:
        msg = f"Failed to access slave ({slave})."
        try:
            result = _call(["ping", "-c", "4", slave])
        except Exception:
            raise Exception(msg)
        if result != 0:
            raise Exception(msg)

        msg = f"Failed to login slave ({slave}) without password."
        try:
            result = _call([
                "ssh", "-o", "NumberOfPasswordPrompts=0", "-o", "StrictHostKeyChecking=yes", f"{slave}",
                "\"/bin/echo\""])
        except Exception:
            raise Exception(msg)
        if result != 0:
            raise Exception(msg)
    _print("Pass.")


def _verify_nfs(args):
    _print("*" * 32)
    _print("Start verify NFS.")
    if not os.path.exists(args.nfs_folder):
        raise Exception(f"Shared NFS folder({args.nfs_folder}) is not existed.")
    for slave in args.slaves:
        temp_folder = os.path.join(args.nfs_folder, uuid.uuid1().hex)
        msg = f"Shared NFS folder ({slave}:{args.nfs_folder}) is not accessed."
        try:
            result = _call(["ssh", slave, f"mkdir {temp_folder}"])
        except Exception:
            raise Exception(msg)
        if result != 0:
            raise Exception(msg)

        try:
            result = _call(["ssh", slave, f"rm -r {temp_folder}"])
        except Exception:
            raise Exception(msg)
        if result != 0:
            raise Exception(msg)
    _print("Pass.")


def _verify_pkg(args):
    _print("*" * 32)
    _print("Start verify packages.")
    # python
    main_output = _check_output([General.python_command, "--version"])
    for slave in args.slaves:
        slave_output = _check_output(["ssh", slave, General.python_command, "--version"])
        if main_output != slave_output:
            raise Exception(f"Python version is different.\nmaster:\n{main_output}\nslave:\n{slave_output}.")
    # main packages
    pkgs = ["noah-vega", "distributed", "torch"]
    for pkg in pkgs:
        main_output = _check_output(["pip3", "show", pkg])
        properties = main_output.split("\n")
        main_version = ""
        for prop in properties:
            if "Version:" in prop:
                main_version = prop
        if main_version == "":
            raise Exception(f"Package ({pkg}) is missing.")
        for slave in args.slaves:
            slave_output = _check_output(["ssh", slave, "pip3", "show", pkg])
            properties = slave_output.split("\n")
            slave_version = ""
            for prop in properties:
                if "Version:" in prop:
                    slave_version = prop
            if main_version != slave_version:
                raise Exception(f"Package is different.\n\nmaster:\n{main_output}\n\nslave:\n{slave_output}.")
    _print("Pass.")


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
        _print("Found existed dask scheduler or dask worker processes.")
        _input = input("Do you want kill dask processes and continue to verify? [Y/n]: ")
        if _input.upper() in ["N", "NO"]:
            _print("Cluster verification canceled.")
            os._exit(0)
        elif _input.upper() not in ["", "Y", "YES"]:
            _print("Input Error.")
            os._exit(0)
        for pid in dask_pids:
            os.kill(int(pid), signal.SIGKILL)
        time.sleep(10)


def _init_dask_scheduler(args):
    _print("Start verify scheduler.")
    global _port
    _port = str(get_available_port())
    try:
        result = _popen(["dask-scheduler", "--port", _port])
    except Exception:
        raise Exception("Failed to start dask scheduler.")
    if not isinstance(result, subprocess.Popen):
        _print("Failed to start dask scheduler.")
        _print("Please run the command in CLI, and resovlue the problems.")
        _print(f"dask-scheduler --port {_port}")
        raise Exception("Failed to start dask scheduler.")
    time.sleep(5)
    _print("Pass.")


def _verfiy_local(args):
    global _port
    _print(f"Start verify local worker, IP:{args.master}, port: {_port}.")
    try:
        result = _popen(["dask-worker", f"{args.master}:{_port}"])
    except Exception:
        raise Exception("Can not start local dask-worker.")
    if not isinstance(result, subprocess.Popen):
        raise Exception("Can not start local dask-worker.")
    time.sleep(5)
    _print("Pass.")

    _print("Test local dask Client.")
    cmd = f"{General.python_command} -c \"from dask.distributed import Client;"\
          f"client=Client('{args.master}:{_port}');client.close()\""
    try:
        result = _call(cmd, shell=True)
    except Exception:
        raise Exception("Can not start local dask client.")
    if result != 0:
        raise Exception("Can not start local dask client.")
    _print("Pass.")


def _verify_client(args):
    global _port
    _print("Start verify slave workers.")
    for slave in args.slaves:
        _print(f"Start verify slave({slave}) worker.")
        try:
            result = _popen(["ssh", slave, f"{shutil.which('dask-worker')} {args.master}:{_port}"])
        except Exception:
            raise Exception(f"Can not start slave({slave}) dask-worker.")
        if not isinstance(result, subprocess.Popen):
            raise Exception(f"Can not start slave({slave}) dask-worker.")
        time.sleep(5)
        _print("Pass.")

        _print(f"Test slave({slave}) dask Client.")
        cmd = f"{General.python_command} -c \"from dask.distributed import Client;"\
              f"client=Client('{args.master}:{_port}');client.close()\""
        try:
            result = _call(cmd, shell=True, env=os.environ)
        except Exception:
            raise Exception(f"Can not start slave({slave}) dask client.")
        if result != 0:
            raise Exception(f"Can not start slave({slave}) dask client.")
        time.sleep(5)
        _print("Pass.")
    _print("Pass.")


def _stop_dask_scheduler(args):
    global _port
    _print("Start stop scheduler.")
    client = Client(f"{args.master}:{_port}")
    try:
        client.shutdown()
        client.close()
        del client
        time.sleep(8)
    except Exception:
        _print("Failed to stop scheduler, please stop it manually.")


def _verify_dask(args):
    _print("*" * 32)
    # _kill_existed_dask(args)
    _init_dask_scheduler(args)
    _verfiy_local(args)
    _verify_client(args)
    _stop_dask_scheduler(args)
    _print("Pass.")


def _verify_cluster():
    args = _parse_args()
    global _json
    _json = args.json
    try:
        _verify_ip(args)
        _verify_nfs(args)
        _verify_pkg(args)
        _verify_dask(args)
        _print("All cluster check items have passed.")
        if args.json:
            print(json.dumps({"status": "success"}, indent=4))
    except Exception as e:
        _print("")
        _print(f"Exception:\n\n{str(e)}")
        if args.json:
            print(json.dumps({"status": "error", "message": str(e)}, indent=4))


if __name__ == "__main__":
    _verify_cluster()
