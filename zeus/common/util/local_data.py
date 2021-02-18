# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE
"""Make local file for record train information in detail."""

import os
import csv
from datetime import datetime
import yaml

from .evaluate_xt import (
    TRAIN_RECORD_CSV,
    TRAIN_CONFIG_YAML,
    DEFAULT_FIELDS,
)
from .default_xt import XtBenchmarkConf as XBConf
from .benchmark_data import Data


__all__ = [
    "LocalDataWriter",
]


def open_file(file_path, open_type):
    """Need close by hand."""
    if file_path.startswith("s3://"):
        import moxing as mox

        ret_handle = mox.file.File(file_path, open_type)
    else:
        ret_handle = open(file_path, open_type)
    return ret_handle


class LocalDataWriter(Data):
    """Make local file to store train record data."""

    def __init__(self, workspace):
        super(LocalDataWriter, self).__init__()
        self.workspace = workspace

        self._config_yaml = os.path.join(self.workspace, TRAIN_CONFIG_YAML)
        self._record_csv = os.path.join(self.workspace, TRAIN_RECORD_CSV)
        self._record_fields = DEFAULT_FIELDS

        self._record_file_open = open_file(self._record_csv, "w+")

        self._record_writer = csv.DictWriter(
            self._record_file_open, fieldnames=self._record_fields
        )

    def get_workspace(self):
        """Get workspace."""
        return self.workspace

    def get_csv(self):
        """Get csv."""
        return self._record_csv

    def insert_records(self, to_record):
        """Insert records."""
        if self._record_writer:
            if isinstance(to_record, list):
                for _item in to_record:
                    once_record = {
                        _k: _v for _k, _v in _item.items() if _k in self._record_fields
                    }
                    self._record_writer.writerow(once_record)
            elif isinstance(to_record, dict):
                once_record = {
                    _k: _v for _k, _v in to_record.items() if _k in self._record_fields
                }
                self._record_writer.writerow(once_record)

            self._record_file_open.flush()
        else:
            print("writing file is closing.")

    def add_new_train_event(self, bm_info):
        """
        Record once new train event.

        e.g, training with new config
        and, hold the train uuid by self.

        Args:
        ----
            bm_info {dict} -- the information of once train configs
        """
        train_conf_elem = bm_info
        eval_gap = bm_info.get("bm_eval", {}).get("gap", XBConf.default_train_interval_per_eval)

        train_conf_elem.update(
            {
                "start_time": datetime.now(),
                "eval_gap": eval_gap,
                "config_yaml": self._config_yaml,
                "record_csv": self._record_csv,
                "workspace": self.workspace,
            }
        )

        con_file_handler = open_file(self._config_yaml, "w")
        yaml.dump(train_conf_elem, con_file_handler)
        con_file_handler.close()

        self._record_writer.writeheader()

    def close(self):
        """Update end time, and close csv file."""
        con_file_handler = open_file(self._config_yaml, "r")
        raw_config_val = yaml.safe_load(con_file_handler)
        con_file_handler.close()

        raw_config_val.update({"end_time": datetime.now()})
        new_file_handler = open_file(self._config_yaml, "w")
        yaml.dump(raw_config_val, new_file_handler)
        new_file_handler.close()

        self._record_file_open.close()
