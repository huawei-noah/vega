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
import json
import time
from datetime import datetime
from vega.common import Status, JsonEncoder, DatatimeFormatString, argment_parser
from vega.tools.query_process import get_pid


__all__ = ["query_progress"]
time_limit = 180


def _parse_args(desc):
    parser = argment_parser(desc)
    parser.add_argument("-t", "--task_id", type=str, required=True,
                        help="vega application task id")
    parser.add_argument("-r", "--root_path", type=str, required=True,
                        help="root path where vega application is running")
    args = parser.parse_args()
    return args


def _get_report_path(root_path, task_id):
    task_path = os.path.join(root_path, task_id)
    report_path = os.path.join(task_path, "output/reports.json")
    return report_path


def _load_report(report_path):
    try:
        with open(report_path, "r") as f:
            return json.load(f)
    except Exception:
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
            if "finished_epochs" in step and step["finished_epochs"] != 0:
                start_time = datetime.strptime(step["start_time"], DatatimeFormatString)
                delta = datetime.now() - start_time
                delta = delta * (step["num_epochs"] - step["finished_epochs"]) / step["finished_epochs"]
                step["estimated_end_time"] = datetime.now() + delta
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


def query_progress():
    """Query vega progress."""
    args = _parse_args("Query Vega progress.")
    is_running = get_pid(args.task_id)
    report_path = _get_report_path(args.root_path, args.task_id)

    if not os.path.exists(report_path):
        if is_running is None:
            run_time = time.time() - is_running.create_time()
            if run_time > 0 and run_time < time_limit:
                return json.dumps({
                    "status": Status.error,
                    "message": "The task is being created, please query again."
                }, cls=JsonEncoder, indent=4)
            else:
                return json.dumps({
                    "status": Status.error,
                    "message": "The task does not exist, please check root path and task id."
                }, cls=JsonEncoder, indent=4)
        else:
            return json.dumps({
                "status": Status.initializing,
            }, cls=JsonEncoder, indent=4)

    report = _load_report(report_path)
    if not report:
        return json.dumps({
            "status": Status.error,
            "message": "Failed to read report file."
        }, cls=JsonEncoder, indent=4)

    progress = _parse_report(report)
    progress = _statistic_progress(progress)
    if progress["status"] == Status.running and not is_running:
        progress["status"] = Status.stopped

    return json.dumps(progress, cls=JsonEncoder, indent=4)


def print_progress():
    """Print progress."""
    print(query_progress())


if __name__ == "__main__":
    print_progress()
