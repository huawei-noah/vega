# -*- coding: utf-8 -*-

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

"""Query vega process."""

import logging
import json
import time
import psutil
from psutil import _pprint_secs
from vega.common import MessageServer, MessageClient, argment_parser
from vega import security
from vega.common.general import General


__all__ = [
    "query_task_info", "get_pid", "is_vega_process", "get_vega_pids",
    "query_process", "query_processes", "print_process"
]


def _parse_args(desc):
    parser = argment_parser(desc)
    parser.add_argument("-j", "--json", action='store_true',
                        help="return json format string")
    parser = security.add_args(parser)
    args = parser.parse_args()
    return args


def get_vega_pids():
    """Get vega pids."""
    pids = psutil.pids()
    vega_pids = []
    for pid in pids:
        if is_vega_process(pid):
            try:
                p = psutil.Process(pid)
            except Exception:
                logging.debug('Failed to get obj of pid.')
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


def get_task_id_path_port(pid):
    """Get task id."""
    try:
        p = psutil.Process(pid)
        for connection in p.connections():
            port = connection.laddr.port
            ip = connection.laddr.ip
            if port in range(MessageServer().min_port, MessageServer().max_port):
                client = MessageClient(ip=ip, port=port, timeout=1)
                result = client.send(action="query_task_info")
                if isinstance(result, dict) and "task_id" in result:
                    return result.get("task_id"), result.get("base_path"), ip, port, None
        return None, None, None, None, "Unknown"
    except Exception as e:
        return None, None, None, None, str(e)


def get_pid(task_id):
    """Get process id."""
    processes = query_processes()
    for process in processes.values():
        if "task_id" in process and task_id == process["task_id"]:
            return process["PID"]
    return None


def is_vega_process(pid):
    """Is it vega process."""
    try:
        p = psutil.Process(pid)
        if p.name().startswith("vega-main"):
            return True
    except Exception:
        return False
    return False


def _print_processes_info(processes):
    if processes:
        print("Vega processes:")
        for id in processes:
            print("{}:".format(id))
            process = processes[id]
            print_process(process)
            if "task_id" in process and process["task_id"] != "Unknown":
                _pid = process["PID"]
                _task_id = process["task_id"]
                _cwd = process["cwd"]
                _base_path = process["base_path"]
        if "_pid" in locals():
            print("")
            if _task_id != "Unknown":
                print("Query progress:")
                print(f"    vega-progress -t {_task_id} -r {_base_path}")
                print("")
            print("Kill process:")
            print(f"    vega-kill -p {_pid}")
            if _task_id != "Unknown":
                print(f"    vega-kill -t {_task_id}")
            print("")
    else:
        print("The Vega main program is not found.")


def print_process(process):
    """Print process info."""
    if "task_id" in process:
        print("       PID: {}".format(process["PID"]))
        print("   task id: {}".format(process["task_id"]))
        print("       cwd: {}".format(process["cwd"]))
        print("      user: {}".format(process["user"]))
        print("  start at: {}".format(process["create_time"]))
        print("   cmdline: {}".format(process["cmdline"]))
    else:
        print("       PID: {}".format(process["PID"]))
        print("   message: {}".format(process["message"]))


def query_process(pid):
    """Query process info."""
    try:
        p = psutil.Process(pid)
        (task_id, base_path, ip, port, msg) = get_task_id_path_port(pid)
        return {
            "PID": pid,
            "cmdline": p.cmdline()[2:],
            "create_time": _pprint_secs(p.create_time()),
            "cwd": p.cwd(),
            "task_id": task_id if task_id is not None else f"error: {msg}",
            "base_path": base_path if base_path is not None else "Unknown",
            "user": p.username(),
            "ip": ip,
            "port": port,
            "running_seconds": int(time.time() - p.create_time()),
        }
    except Exception as e:
        return {
            "PID": pid,
            "message": str(e),
        }


def query_task_info(task_id):
    """Query task info."""
    pids = get_vega_pids()
    if pids:
        for id, pid in enumerate(pids):
            info = query_process(pid)
            if isinstance(info, dict) and info.get("task_id", None) == task_id:
                return info
    return None


def query_processes():
    """Query all process."""
    pids = get_vega_pids()
    infos = {}
    if pids:
        for id, pid in enumerate(pids):
            infos[str(id)] = query_process(pid)
    return infos


def main():
    """Print all processes."""
    args = _parse_args("Quey Vega processes.")
    if args.security:
        if not security.load_config("client"):
            print("If you run vega in normal mode, use parameter '-s'.")
            print("For more parameters: vega-process --help")
            return
    General.security = args.security
    processes = query_processes()
    if args.json:
        print(json.dumps(processes, indent=4))
    else:
        _print_processes_info(processes)


if __name__ == "__main__":
    main()
