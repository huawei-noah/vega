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

"""Distributed worker for training and evaluating.

Distributed worker is the basic class of TrainWorker and EvaluatorWork,
it loads the pickle file into worker from master, and run the train_process
function of each distributed worker on local node, it also has the function
of timeout, killing the worker process which exceeds setting time.
"""
from vega.common import TaskOps
from vega.common.general import General


class DistributedWorker(TaskOps):
    """Class of Distributed Worker.

    This is a distributed worker used to load worker's pickle file,
    and run the process of training and evaluating.

    :param args: arguments from user config file
    :type args: dict or Config, default to None
    """

    __worker_id__ = 0

    def __init__(self, args=None):
        """Init DistributedWorker."""
        super(DistributedWorker, self).__init__()
        # privates
        DistributedWorker.__worker_id__ += 1
        self._worker_id = DistributedWorker.__worker_id__
        # publics
        self.rank = 0
        self.world_size = 1
        self.worker_addr = ""
        self.worker_nccl_port = 16666
        self.timeout = int(float(General.worker.timeout) * 60 * 60)
        return

    @property
    def worker_id(self):
        """Property: worker_id."""
        return self._worker_id

    @worker_id.setter
    def worker_id(self, value):
        """Setter: set worker_id with value.

        :param value: worker id
        :type value: int
        """
        self._worker_id = value

    def train_process(self):
        """Abstract base function for DistributedWorker to do the train process."""
        raise NotImplementedError
