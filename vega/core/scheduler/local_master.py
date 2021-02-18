# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The LocalMaster's method is same as Master, and the class is used on single node."""
import os

from zeus.trainer.utils import WorkerTypes
from zeus.common.general import General


class LocalMaster(object):
    """The Master's method is same as Master."""

    def __init__(self, update_func=None):
        """Init master."""
        self.cfg = General
        self.step_name = None
        self.worker_id = None
        self.update_func = update_func
        if os.environ['DEVICE_CATEGORY'] == 'NPU':
            os.environ['RANK_SIZE'] = '1'
            os.environ.pop('RANK_TABLE_FILE', None)

    def run(self, worker, evaluator=None):
        """Run a worker, call the worker's train_prcess() method.

        :param worker: a worker.
        :type worker: object that the class was inherited from DistributedWorker.

        """
        if worker is None:
            return
        self.step_name = worker.step_name
        self.worker_id = worker.worker_id
        if worker.worker_type == WorkerTypes.EVALUATOR:
            for sub_worker in worker.sub_worker_list:
                sub_worker.train_process()
        elif worker.worker_type == WorkerTypes.HAVA_D_EVALUATOR:
            pass
        else:
            worker.train_process()
        if evaluator:
            if evaluator.worker_type == WorkerTypes.EVALUATOR:
                for sub_worker in evaluator.sub_worker_list:
                    sub_worker.train_process()
            elif evaluator.worker_type == WorkerTypes.HAVA_D_EVALUATOR:
                pass
        self._update(self.step_name, self.worker_id)

    def _update(self, step_name, worker_id):
        if not self.update_func:
            return
        if self.update_func.__code__.co_varnames.index("step_name") == 1:
            self.update_func(step_name, worker_id)
        else:
            self.update_func({"step_name": step_name, "worker_id": worker_id})

    def join(self):
        """Return immediately."""
        return

    def close_client(self):
        """Close cluster client, implement the interface without actually closing."""
        pass

    def shutdown(self):
        """Shut down the cluster, implement the interface without actually shutting down."""
        pass
