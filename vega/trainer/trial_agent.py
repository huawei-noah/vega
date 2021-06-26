# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Base Trainer."""

import os
import logging
import pickle
import vega
from vega.common import init_log
from vega.common.task_ops import TaskOps
from vega.common.general import General
from vega.report.report_client import ReportClient

logger = logging.getLogger(__name__)


class TrialAgent(object):
    """Trial."""

    def __init__(self):
        self._load_config()
        vega.set_backend(General.backend, General.device_category)
        init_log(level=General.logger.level,
                 log_file=f"{General.step_name}_worker_{self.worker_id}.log",
                 log_path=TaskOps().local_log_path)
        self.report_client = ReportClient()

    def _load_config(self):
        _file = os.path.join(os.path.curdir, ".trial")
        with open(_file, "rb") as f:
            data = pickle.load(f)
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
