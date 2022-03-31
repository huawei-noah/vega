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

"""Report."""
import json
import logging
import os
import traceback
from pickle import HIGHEST_PROTOCOL
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

            def update_each_step_info(step_name):
                if step_name not in self.steps:
                    self.steps[step_name] = {}
                for key in kwargs:
                    if key == "step_name":
                        self.steps[step_name][key] = step_name
                    elif key in ["start_time", "end_time", "status",
                               "message", "num_epochs", "num_models", "best_models"]:
                        self.steps[step_name][key] = kwargs[key]
                    else:
                        logger.warn("Invilid step info {}:{}".format(key, kwargs[key]))

            if isinstance(step_name, list):
                for step in step_name:
                    update_each_step_info(step)
            else:
                update_each_step_info(step_name)
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
        except Exception as e:
            logging.warning(f"Failed to save report, message: {e}")
            logging.debug(traceback.format_exc())

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
        except Exception as e:
            logging.warning(f"Failed to get report, message: {e}")
            logging.debug(traceback.format_exc())

    def pickle_report(self, records, report_instance):
        """Pickle report to `.reports`."""
        try:
            _file = os.path.join(TaskOps().step_path, ".reports")
            _dump_data = [records, report_instance]
            FileOps.dump_pickle(_dump_data, _file, protocol=HIGHEST_PROTOCOL)
        except Exception as e:
            logging.warning(f"Failed to pickle report, message: {e}")
            logging.debug(traceback.format_exc())
