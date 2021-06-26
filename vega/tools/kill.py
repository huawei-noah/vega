# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Kill vega progress."""

import os
import signal
import psutil
import time
from vega.common import argment_parser
from vega.tools.query_process import query_process, get_pid, query_processes, get_vega_pids, print_process


def _parse_args(desc):
    parser = argment_parser(desc)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--pid", type=int,
                       help="kill Vega main process based on the specified process ID")
    group.add_argument("-t", "--task_id", type=str,
                       help="kill Vega main process based on the specified Vega application task ID")
    group.add_argument("-a", "--all", action='store_true',
                       help="kill all Vega main process")
    group.add_argument("-f", "--force", action='store_true',
                       help="Forcibly kill all Vega-related processes even if the main process does not exist")
    args = parser.parse_args()
    return args


def _kill_vega_process(pid):
    if not psutil.pid_exists(pid):
        print("The Vega process {} does not exist.".format(pid))
        return
    if pid not in get_vega_pids():
        print("Process {} is not the main Vega process.".format(pid))
        return
    print_process(query_process(pid))
    print("")
    _input = input("Do you want kill vega processes? [Y/n]: ")
    if _input.upper() in ["N", "NO"]:
        print("Operation was cancelled.")
        os._exit(0)

    spids = _get_sub_processes(pid)
    print("Start to kill Vega process {}.".format(pid))
    try:
        os.kill(pid, signal.SIGINT)
    except Exception:
        pass
    _wait(3)
    spids.append(pid)
    not_stoped = _check_exited(spids)
    for pid in not_stoped:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    _wait(5)
    print("")
    not_stoped = _check_exited(not_stoped)
    if _check_exited(not_stoped):
        print("Warning: The following processes do not exit completely.")
        print(not_stoped)
    else:
        print("All Vega processes have been killed.")


def _kill_vega_process_by_task_id(task_id):
    pid = get_pid(task_id)
    if not pid:
        print("Task ID {} is not the task ID of a Vega process.".format(task_id))
        return
    _kill_vega_process(pid)


def _kill_all_vega_process():
    pids = get_vega_pids()
    if not pids:
        print("The Vega main program is not found.")
        return

    print("Vega processes:")
    for key, value in query_processes().items():
        print("{}:".format(key))
        print_process(value)
    print("")
    _input = input("Do you want kill all vega processes? [Y/n]: ")
    if _input.upper() in ["N", "NO"]:
        print("Operation was cancelled.")
        os._exit(0)

    all_spids = []
    all_spids.extend(pids)
    for pid in pids:
        spids = _get_sub_processes(pid)
        all_spids.extend(spids)
        print("Start to kill the Vega process {}".format(pid))
        try:
            os.kill(pid, signal.SIGINT)
        except Exception:
            pass
    _wait(3)
    not_stoped = _check_exited(all_spids)
    for pid in not_stoped:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    _wait(5)
    print("")
    not_stoped = _check_exited(not_stoped)
    if _check_exited(not_stoped):
        print("Warning: The following processes do not exit completely.")
        print(not_stoped)
    else:
        print("All Vega processes have been killed.")


def _get_sub_processes(pid, cpids=[]):
    p = psutil.Process(pid)
    for cp in p.children():
        cpid = cp.pid
        cpids.append(cpid)
        try:
            _get_sub_processes(cpid, cpids)
        except Exception:
            pass
    return cpids


def _force_kill():
    vega_pids = _get_all_related_processes()
    if not vega_pids:
        print("No Vega-releted progress found.")
        return

    _input = input("Do you want kill all Vega-related processes? [Y/n]: ")
    if _input.upper() in ["N", "NO"]:
        print("Operation was cancelled.")
        os._exit(0)

    vega_pids = _get_all_related_processes()
    print("Start to kill all Vega-related processes.")
    for pid in vega_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    _wait(5)
    print("")
    not_stoped = _check_exited(vega_pids)
    if not_stoped:
        print("Warning: The following processes do not exit completely.")
        print(not_stoped)
    else:
        print("All Vega-related processes have been killed.")


def _get_all_related_processes():
    pids = psutil.pids()
    vega_pids = []
    for pid in pids:
        try:
            p = psutil.Process(pid)
        except Exception:
            continue
        if p.name() in ["vega", "dask-scheduler", "dask-worker", "vega-main"]:
            vega_pids.append(pid)
            vega_pids.extend(_get_sub_processes(pid))
            continue
        cmd = " ".join(p.cmdline())
        if "/bin/vega-kill" in cmd or "/bin/vega-process" in cmd or "/bin/vega-progress" in cmd:
            continue
        if "vega.tools.run_pipeline" in cmd or "vega.trainer.deserialize" in cmd or "/bin/vega" in cmd:
            vega_pids.append(pid)
            vega_pids.extend(_get_sub_processes(pid))
            continue
    vega_pids = set(vega_pids)
    return vega_pids


def _check_exited(pids):
    not_killed = []
    for pid in pids:
        if psutil.pid_exists(pid):
            not_killed.append(pid)
    return not_killed


def _wait(seconds):
    for _ in range(seconds * 2):
        print("*", end="", flush=True)
        time.sleep(0.5)


def _kill():
    args = _parse_args("Kill Vega processes.")
    if args.pid:
        _kill_vega_process(args.pid)
    elif args.task_id:
        _kill_vega_process_by_task_id(args.task_id)
    elif args.all:
        _kill_all_vega_process()
    elif args.force:
        _force_kill()


if __name__ == "__main__":
    _kill()
