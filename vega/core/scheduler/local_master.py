# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The LocalMaster's method is same as Master, and the class is used on single node."""
import copy
from vega.core.common.user_config import UserConfig
from ..trainer.utils import WorkerTypes


class LocalMaster(object):
    """The Master's method is same as Master."""

    def __init__(self):
        """Init master."""
        self.cfg = copy.deepcopy(UserConfig().data.general)
        self.step_name = None
        self.worker_id = None

    def run(self, worker):
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

    def join(self):
        """Return immediately."""
        return

    def pop_finished_worker(self, train_worker=True):
        """Pop saved worker id and step name.

        :return: the finished worker info, include step_name and worker_id.
            eg. {"step_name":"round1", "worker_id":1}
        :rtype: dict or None

        """
        if self.worker_id is None:
            return None
        worker_id = self.worker_id
        step_name = self.step_name
        self.worker_id = None
        self.step_name = None
        return {"step_name": step_name, "worker_id": worker_id}

    def pop_finished_train_worker(self):
        """Pop saved worker id and step name.

        :return: the finished worker info, include step_name and worker_id.
            eg. {"step_name":"round1", "worker_id":1}
        :rtype: dict or None

        """
        return self.pop_finished_worker(train_worker=True)

    def pop_finished_evaluate_worker(self):
        """Pop saved worker id and step_name.

        :return: the finished worker info, include step_name and worker_id.
            eg. {"step_name":"round1", "worker_id":1}
        :rtype: dict or None

        """
        return self.pop_finished_worker(train_worker=False)

    def pop_all_finished_train_worker(self):
        """Pop saved worker id and step name.

        :return: a finished worker info list.
        :rtype: list of dict

        """
        worker_info_list = []
        finished_train_worker_info = self.pop_finished_train_worker()
        if finished_train_worker_info is not None:
            worker_info_list.append(finished_train_worker_info)
        return worker_info_list

    def pop_all_finished_evaluate_worker(self):
        """Pop saved worker id and step name.

        :return: a finished worker info list.
        :rtype: list of dict

        """
        return self.pop_all_finished_train_worker()

    def close_cluster(self):
        """Close all distributed cluster, implement the interface without actually closing."""
        pass

    def close_client(self):
        """Close cluster client, implement the interface without actually closing."""
        pass

    @staticmethod
    def shutdown():
        """Shut down the cluster, implement the interface without actually shutting down."""
        pass
