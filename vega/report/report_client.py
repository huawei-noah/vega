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
from datetime import datetime
from vega.common import FileOps, TaskOps
from vega.common.utils import remove_np_value
from vega.common import MessageClient
from vega.common import General, Status, JsonEncoder
from .record import ReportRecord


logger = logging.getLogger(__name__)


class ReportClient(object):
    """Report class to save all records and update records to share memory."""

    def __init__(self):
        self.client = MessageClient(ip=General.cluster.master_ip, port=General.message_port)

    def update(self, step_name, worker_id, **kwargs):
        """Update record."""
        if not isinstance(kwargs, dict):
            kwargs = {}
        kwargs["step_name"] = step_name
        kwargs["worker_id"] = worker_id
        kwargs = json.loads(json.dumps(kwargs, cls=JsonEncoder))
        result = self.client.send(action="update_record", data=kwargs)
        if not isinstance(result, dict) or "result" not in result or result["result"] != "success":
            raise Exception(f"Failed to update record: {result}")
        record = ReportRecord().load_dict(result["data"])
        self._save_worker_record(record.to_dict())
        return record

    def set_finished(self, step_name, worker_id):
        """Set record finished."""
        kwargs = {}
        kwargs["step_name"] = step_name
        kwargs["worker_id"] = worker_id
        kwargs["end_time"] = datetime.now()
        kwargs["status"] = Status.finished
        kwargs = json.loads(json.dumps(kwargs, cls=JsonEncoder))
        result = self.client.send(action="update_record", data=kwargs)
        if not isinstance(result, dict) or "result" not in result or result["result"] != "success":
            raise Exception(f"Failed to set finished: {result}")
        record = ReportRecord().load_dict(result["data"])
        self._save_worker_record(record.to_dict())
        return record

    def request(self, action, **kwargs):
        """Set record finished."""
        kwargs = json.loads(json.dumps(kwargs, cls=JsonEncoder))
        return self.client.send(action=action, data=kwargs)

    def get_record(self, step_name, worker_id):
        """Get value from Shared Memory."""
        result = self.client.send(action="get_record", data={"step_name": step_name, "worker_id": worker_id})
        if not isinstance(result, dict) or "result" not in result or result["result"] != "success":
            raise Exception(f"Failed to get record: {result}")
        return ReportRecord().load_dict(result["data"])

    def _save_worker_record(self, record):
        step_name = record.get('step_name')
        worker_id = record.get('worker_id')
        _path = TaskOps().get_local_worker_path(step_name, worker_id)
        for record_name in ["desc", "hps", "performance"]:
            _file_name = None
            _file = None
            record_value = remove_np_value(record.get(record_name))
            if record_value is None:
                if record_name == "desc":
                    record_value = {}
                else:
                    continue
            _file = None
            try:
                # for cars/darts save multi-desc
                if isinstance(record_value, list) and record_name == "desc":
                    for idx, value in enumerate(record_value):
                        _file_name = "desc_{}.json".format(idx)
                        _file = FileOps.join_path(_path, _file_name)
                        with open(_file, "w") as f:
                            json.dump(value, f)
                else:
                    if 'multi_task' in record:
                        worker_id = record.get('multi_task') if record.get('multi_task') is not None else worker_id
                    _file_name = None
                    if record_name == "desc":
                        _file_name = "desc_{}.json".format(worker_id)
                    if record_name == "hps":
                        _file_name = "hps_{}.json".format(worker_id)
                    if record_name == "performance":
                        _file_name = "performance_{}.json".format(worker_id)
                    _file = FileOps.join_path(_path, _file_name)
                    with open(_file, "w") as f:
                        json.dump(record_value, f)
            except Exception as ex:
                logger.error("Failed to save {}, file={}, desc={}, msg={}".format(
                    record_name, _file, record_value, str(ex)))
