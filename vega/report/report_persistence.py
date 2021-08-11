# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Report."""
import json
import logging
import os
import traceback
import pickle
from vega.common import FileOps, TaskOps, JsonEncoder, Status


logger = logging.getLogger(__name__)


class ReportPersistence(object):
    """Save report to file (reports.json)."""

    def __init__(self):
        """Initialize object."""
        self.step_names = []
        self.steps = {}

    def set_step_names(self, step_names):
        """Set name of steps."""
        self.step_names = step_names

    def update_step_info(self, **kwargs):
        """Update step info."""
        if "step_name" in kwargs:
            step_name = kwargs["step_name"]
            if step_name not in self.steps:
                self.steps[step_name] = {}
            for key in kwargs:
                if key in ["step_name", "start_time", "end_time", "status", "message", "num_epochs", "num_models"]:
                    self.steps[step_name][key] = kwargs[key]
                else:
                    logger.warn("Invilid step info {}:{}".format(key, kwargs[key]))
        else:
            logger.warn("Invilid step info: {}.".format(kwargs))

    def save_report(self, records):
        """Save report to `reports.json`."""
        try:
            _file = FileOps.join_path(TaskOps().local_output_path, "reports.json")
            FileOps.make_base_dir(_file)
            data = self.get_report(records)
            with open(_file, "w") as f:
                json.dump(data, f, indent=4, cls=JsonEncoder)
        except Exception:
            logging.warning(traceback.format_exc())

    def get_report(self, records):
        """Save report to `reports.json`."""
        try:
            data = {"_steps_": []}
            for step in self.step_names:
                if step in self.steps:
                    data["_steps_"].append(self.steps[step])
                else:
                    data["_steps_"].append({
                        "step_name": step,
                        "status": Status.unstarted
                    })
            for record in records:
                if record.step_name in data:
                    data[record.step_name].append(record.to_dict())
                else:
                    data[record.step_name] = [record.to_dict()]
            return data
        except Exception:
            logging.warning(traceback.format_exc())

    def pickle_report(self, records, report_instance):
        """Pickle report to `.reports`."""
        try:
            _file = os.path.join(TaskOps().step_path, ".reports")
            _dump_data = [records, report_instance]
            with open(_file, "wb") as f:
                pickle.dump(_dump_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            logging.warning(traceback.format_exc())
