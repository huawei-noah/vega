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
from zeus.common.file_ops import FileOps
from zeus.common.task_ops import TaskOps
from zeus.report.share_memory import ShareMemory
from zeus.common.utils import remove_np_value
from .record import ReportRecord


class ReportClient(object):
    """Report class to save all records and broadcast records to share memory."""

    @classmethod
    def broadcast(cls, record):
        """Broadcast one record to Shared Memory."""
        if not record:
            logging.warning("Broadcast Record is None.")
            return
        ShareMemory("{}.{}".format(record.step_name, record.worker_id)).put(record.serialize())
        cls._save_worker_record(record.serialize())

    @classmethod
    def get_record(cls, step_name, worker_id):
        """Get value from Shared Memory."""
        value = ShareMemory("{}.{}".format(step_name, worker_id)).get()
        if value:
            record = ReportRecord().from_dict(value)
        else:
            record = ReportRecord(step_name, worker_id)
        return record

    @classmethod
    def close(cls, step_name, worker_id):
        """Clear Shared Memory."""
        ShareMemory("{}.{}".format(step_name, worker_id)).close()

    @classmethod
    def _save_worker_record(cls, record):
        step_name = record.get('step_name')
        worker_id = record.get('worker_id')
        _path = TaskOps().get_local_worker_path(step_name, worker_id)
        for record_name in ["desc", "hps", "performance"]:
            _file_name = None
            _file = None
            record_value = remove_np_value(record.get(record_name))
            if not record_value:
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
                logging.error("Failed to save {}, file={}, desc={}, msg={}".format(
                    record_name, _file, record_value, str(ex)))
