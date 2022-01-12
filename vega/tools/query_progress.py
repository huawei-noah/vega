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

"""Inference of vega model."""

import os
import json
import time
from datetime import datetime
from vega.common import Status, JsonEncoder, DatatimeFormatString, argment_parser
from vega.tools.query_process import query_task_info
from vega.common import MessageClient
from vega import security
from vega.common.general import General


__all__ = ["query_progress"]
error_message = ""


def _parse_args(desc):
    parser = argment_parser(desc)
    parser.add_argument("-t", "--task_id", type=str, required=True,
                        help="vega application task id")
    parser.add_argument("-r", "--root_path", type=str, required=True,
                        help="root path where vega application is running")
    parser = security.args.add_args(parser)
    args = parser.parse_args()
    security.args.check_args(args)
    return args


def _get_report_path(root_path, task_id):
    task_path = os.path.join(root_path, task_id)
    report_path = os.path.join(task_path, "output/reports.json")
    return report_path


def _load_report(report_path):
    try:
        with open(report_path, "r") as f:
            return json.load(f)
    except Exception as e:
        global error_message
        error_message = str(e)
        return None


def _parse_report(report):
    if "_steps_" not in report:
        return {
            "status": Status.error,
            "message": "Invalid report file."
        }

    progress = {
        "steps": report["_steps_"]
    }

    model_keys = [
        "worker_id", "status", "message", "current_epoch", "num_epochs",
        "start_time", "end_time", "model_path", "performance"
    ]

    for step in progress["steps"]:
        step_name = step["step_name"]
        if step_name not in report:
            continue
        step["models"] = report[step_name]
        for model in step["models"]:
            keys = list(model.keys())
            for key in keys:
                if key not in model_keys:
                    model.pop(key)
    return progress


def _statistic_progress(progress):
    # count epochs and models
    for step in progress["steps"]:
        finished_models = 0
        finished_epochs = 0
        if "models" not in step:
            continue
        for model in step["models"]:
            if model["status"] in [Status.finished.value, Status.finished]:
                finished_models += 1
                finished_epochs += model["current_epoch"]
            else:
                current_epoch = max((model["current_epoch"] - 1), 0) if "current_epoch" in model else 0
                finished_epochs += current_epoch
        step["finished_models"] = finished_models
        step["finished_epochs"] = finished_epochs
    # calc time
    for step in progress["steps"]:
        step["estimated_end_time"] = None
        if step["status"] == Status.running.value:
            if "finished_epochs" in step and step["finished_epochs"] != 0 and "num_epochs" in step:
                start_time = datetime.strptime(step["start_time"], DatatimeFormatString)
                delta = datetime.now() - start_time
                delta = delta * (step["num_epochs"] - step["finished_epochs"]) / step["finished_epochs"]
                step["estimated_end_time"] = datetime.now() + delta
            else:
                step["estimated_end_time"] = "0000-00-00 00:00:00"
    # count status
    all_finished = True
    progress["status"] = Status.running
    for step in progress["steps"]:
        if step["status"] in [Status.error.value, Status.error]:
            progress["status"] = Status.error
            progress["message"] = step["message"]
            all_finished = False
            break
        if step["status"] not in [Status.finished.value, Status.finished]:
            all_finished = False
            break
    if all_finished:
        progress["status"] = Status.finished

    return progress


def _query_report(task_info):
    """Get task id."""
    try:
        port = task_info["port"]
        ip = task_info["ip"]
        client = MessageClient(ip=ip, port=port, timeout=1)
        return client.send(action="query_report")
    except Exception as e:
        global error_message
        error_message = str(e)
        return None


def query_progress(args, times=0):
    """Query vega progress."""
    task_info = query_task_info(args.task_id)

    if not task_info:
        report_path = _get_report_path(args.root_path, args.task_id)
        if not os.path.exists(report_path):
            times += 1
            if times <= 3:
                time.sleep(0.5)
                query_progress(times)
            else:
                return json.dumps({
                    "status": Status.error,
                    "message": "The task does not exist, please check root path and task id."
                }, cls=JsonEncoder, indent=4)
        report = _load_report(report_path)
    else:
        report = _query_report(task_info)
    if not report:
        global error_message
        return json.dumps({
            "status": Status.error,
            "message": f"Failed to query progress. {error_message}"
        }, cls=JsonEncoder, indent=4)

    progress = _parse_report(report)
    progress = _statistic_progress(progress)
    if progress["status"] == Status.running and not task_info:
        progress["status"] = Status.stopped

    return json.dumps(progress, cls=JsonEncoder, indent=4)


def main():
    """Print progress."""
    args = _parse_args("Query Vega progress.")
    if args.security:
        if not security.load_config("client"):
            print("If you run vega in normal mode, use parameter '-s'.")
            print("For more parameters: vega-progress --help")
            return
    General.security = args.security
    print(query_progress(args))


if __name__ == "__main__":
    main()
