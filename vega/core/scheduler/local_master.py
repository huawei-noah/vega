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

"""The LocalMaster's method is same as Master, and the class is used on single node."""

import traceback
import logging
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

    def run(self, worker, evaluator=None):
        """Run a worker, call the worker's train_prcess() method.

        :param worker: a worker.
        :type worker: object that the class was inherited from DistributedWorker.

        """
        if worker is None:
            return

        step_name = worker.step_name
        worker_id = worker.worker_id

        if worker.worker_type == WorkerTypes.EVALUATOR and evaluator is None:
            workers = []
            evaluator = worker
        else:
            workers = [worker]

        if evaluator and evaluator.worker_type == WorkerTypes.EVALUATOR:
            for sub_worker in evaluator.sub_worker_list:
                is_device_evaluator = sub_worker.worker_type == WorkerTypes.DeviceEvaluator
                if is_device_evaluator and General.device_evaluate_before_train:
                    workers.insert(0, sub_worker)
                else:
                    workers.append(sub_worker)

        for worker in workers:
            try:
                worker.train_process()
            except Exception as e:
                logging.debug(traceback.format_exc())
                logging.error(f"Failed to run worker, id: {worker.worker_id}, message: {e}")
                if not General.skip_trainer_error:
                    raise e

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
