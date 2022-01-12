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

"""The MasterBase class."""


class MasterBase(object):
    """The Master's method is same as Master."""

    def run(self, worker, evaluator=None):
        """Run a worker, call the worker's train_prcess() method.

        :param worker: a worker.
        :type worker: object that the class was inherited from DistributedWorker.

        """
        pass

    def join(self):
        """Return immediately."""
        pass

    def close(self):
        """Close cluster client, implement the interface without actually closing."""
        pass

    def shutdown(self):
        """Shut down the cluster, implement the interface without actually shutting down."""
        pass
