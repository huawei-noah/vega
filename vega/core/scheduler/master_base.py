# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
