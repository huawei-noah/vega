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
from vega.trainer.utils import WorkerTypes
from vega.common.general import General
from vega.report import ReportClient
from .master_base import MasterBase


class LocalMaster(MasterBase):
    """The Master's method is same as Master."""

    def __init__(self, update_func=None):
        """Init master."""
        self.cfg = General
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

        step_name = worker.step_name
        worker_id = worker.worker_id

        workers = [worker]
        if evaluator and evaluator.worker_type == WorkerTypes.EVALUATOR:
            for sub_worker in evaluator.sub_worker_list:
                if sub_worker.worker_type == WorkerTypes.DeviceEvaluator:
                    workers.insert(0, sub_worker)
                else:
                    workers.append(sub_worker)

        for worker in workers:
            worker.train_process()

        self._update(step_name, worker_id)

    def _update(self, step_name, worker_id):
        # Waiting report thread update all record
        ReportClient().set_finished(step_name, worker_id)
        if not self.update_func:
            return
        if self.update_func.__code__.co_varnames.index("step_name") == 1:
            self.update_func(step_name, worker_id)
        else:
            self.update_func({"step_name": step_name, "worker_id": worker_id})
