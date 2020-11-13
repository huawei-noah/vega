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
"""Make database for record."""


class Data(object):
    """
    Make base class of data structure to store train/test information, for analysis relative performance.

    local database will using sqlite.
    local file will work with numpy & csv
    """

    VERSION = 0.1

    def __init__(self):
        self.base_fields = (
            "env_name",  # rl's environment
            "alg_name",  # algorithm
            "train_index",  # the index of model saved, user define
            "start_time",  # this event start time
            "sample_step",  # the total sample steps used for training,
            "train_loss",
            "train_reward",
            "eval_reward",
            "framework",
            "comments",  # user others' comments
        )

    def insert_records(self, to_record):
        """
        Insert train record.

        Args:
        ----
            to_record:
        """
        raise NotImplementedError

    def get_version(self):
        """Get database version info."""
        return self.VERSION
