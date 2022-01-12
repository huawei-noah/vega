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

"""Base Trainer."""

import os
import vega
from vega.common.general import General
from vega.report.report_client import ReportClient
from vega.common import FileOps


class TrialAgent(object):
    """Trial."""

    def __init__(self):
        self._load_config()
        vega.set_backend(General.backend, General.device_category)
        self.report_client = ReportClient()

    def _load_config(self):
        _file = os.path.join(os.path.curdir, ".trial")
        data = FileOps.load_pickle(_file)
        self.worker_id = data["worker_id"]
        self.model_desc = data["model_desc"]
        self.hps = data["hps"]
        self.epochs = data["epochs"]
        General.from_dict(data["general"])

    def update(self, **kwargs):
        """Update record on server."""
        return self.report_client.update(General.step_name, self.worker_id, **kwargs)

    def request(self, action, **kwargs):
        """Set special requst."""
        if "step_name" not in kwargs:
            kwargs["step_name"] = General.step_name
        if "worker_id" not in kwargs:
            kwargs["worker_id"] = self.worker_id
        return self.report_client.request(action, **kwargs)

    def get_record(self):
        """Get value from Shared Memory."""
        return self.report_client.get_record(General.step_name, self.worker_id)
