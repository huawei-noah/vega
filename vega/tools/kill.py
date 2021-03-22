# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Inference of vega model."""

import os
import signal
import argparse
import psutil
import time
from psutil import _pprint_secs


def _parse_args(desc):
    parser = argparse.ArgumentParser(description=desc)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--list", action='store_true',
                       help="list all Vega main process")
    group.add_argument("-p", "--pid", type=int,
                       help="kill Vega main process based on the specified process ID")
    group.add_argument("-a", "--all", action='store_true',
                       help="kill all Vega main process")
    group.add_argument("-f", "--force", action='store_true',
                       help="Forcibly kill all Vega processes even if the main process does not exist")
    args = parser.parse_args()
    return args


def _list_vega_processes():
    pids = _filter_vega_process()
    if pids:
        for pid in pids:
            print(_format(pid, show_all=True))
            # pids = _format_sub_processes(pid)
            # print("\n".join(pids))
        print("If you want kill vega process, please using '-p <pid>' parameter.")
    else:
        print("The Vega main program is not found.")


def _filter_vega_process():
    pids = psutil.pids()
    vega_pids = []
    for pid in pids:
        if _is_vega_process(pid):
            try:
                p = psutil.Process(pid)
            except Exception:
                continue
            ppid = p.ppid()
            if ppid in [_pid for (_pid, _ppid) in vega_pids]:
                continue
            if pid in [_ppid for (_pid, _ppid) in vega_pids]:
                vega_pids = [(_pid, _ppid) for (_pid, _ppid) in vega_pids if _ppid != pid]
                vega_pids.append((pid, ppid))
                continue
            vega_pids.append((pid, ppid))
    return [_pid for (_pid, _ppid) in vega_pids]


def _is_vega_process(pid):
    try:
        p = psutil.Process(pid)
    except Exception:
        return False
    if p.name() == "vega":
        return True
    if "python" in p.name() and "vega.tools.run_pipeline" in "".join(p.cmdline()):
        return True
    return False


def _format_sub_processes(pid, cpids=[], space=""):
    try:
        p = psutil.Process(pid)
    except Exception:
        return cpids
    space += "  "
    for cp in p.children():
        cpid = cp.pid
        cpids.append("{}+ {}".format(space, _format(cpid)))
        _format_sub_processes(cpid, cpids, space)
    return cpids


def _kill_vega_process(pid):
    if not psutil.pid_exists(pid):
        print("The Vega process {} does not exist.".format(pid))
        return
    if not _is_vega_process(pid):
        print("Process {} is not the main Vega process.".format(pid))
        return
    spids = _get_sub_processes(pid)
    print("Start to kill the Vega process {}.".format(pid))
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
    not_stoped = _check_exited(not_stoped)
    if _check_exited(not_stoped):
        print("Warning: The following processes do not exit completely.")
        print(not_stoped)
    else:
        print("All Vega processes have been killed.")


def _kill_all_vega_process():
    pids = _filter_vega_process()
    if not pids:
        print("The Vega main program is not found.")
        return
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
    not_stoped = _check_exited(not_stoped)
    if _check_exited(not_stoped):
        print("Warning: The following processes do not exit completely.")
        print(not_stoped)
    else:
        print("All Vega processes have been killed.")


def _format(pid, show_all=False):
    try:
        p = psutil.Process(pid)
    except Exception:
        return "Process {} is inaccessible.".format(pid)
    if show_all:
        s = "PID: {}, user: {}, create time: {}, cmd: {}".format(
            pid, p.username(), _pprint_secs(p.create_time()), p.cmdline())
    else:
        s = "PID: {}, cmd: {}".format(
            pid, p.cmdline())
    return s


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
    pids = psutil.pids()
    vega_pids = []
    for pid in pids:
        try:
            p = psutil.Process(pid)
        except Exception:
            continue
        if p.name() in ["vega", "dask-scheduler", "dask-worker"]:
            vega_pids.append(pid)
            vega_pids.extend(_get_sub_processes(pid))
            continue
        cmd = " ".join(p.cmdline())
        if "vega.tools.run_pipeline" in cmd or "zeus.trainer.deserialize" in cmd:
            vega_pids.append(pid)
            vega_pids.extend(_get_sub_processes(pid))
            continue
    vega_pids = set(vega_pids)
    if not vega_pids:
        print("No Vega progress found.")
        return
    print("Start to kill all Vega processes.")
    for pid in vega_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    _wait(5)
    not_stoped = _check_exited(vega_pids)
    if not_stoped:
        print("Warning: The following processes do not exit completely.")
        print(not_stoped)
    else:
        print("All Vega processes have been killed.")


def _check_exited(pids):
    not_killed = []
    for pid in pids:
        if psutil.pid_exists(pid):
            not_killed.append(pid)
    return not_killed


def _kill():
    args = _parse_args("Kill Vega processes.")
    if args.list:
        _list_vega_processes()
    elif args.pid:
        _kill_vega_process(args.pid)
    elif args.all:
        _kill_all_vega_process()
    elif args.force:
        _force_kill()


def _wait(seconds):
    for _ in range(seconds * 2):
        print("*", end="", flush=True)
        time.sleep(0.5)
    print("")


if __name__ == "__main__":
    _kill()
