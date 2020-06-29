# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The main part of the cluster framework.

The Master Class to create, maintain distribute framework and distribute
calculate tasks.
"""
import os
import sys
import copy
import logging
import time
import traceback
from queue import Queue
from ..trainer import utils
from .distribution import ClusterDaskDistributor, LocalDistributor
from vega.core.common import UserConfig, TaskOps
from vega.core.common.consts import ClusterMode
from .local_master import LocalMaster
from .worker_env import WorkerEnv
from .dask_env import DaskEnv


class Master(object):
    """The Master Class is to create, maintain distribute framework and distribute calculate tasks.

    :param argparse.ArgumentParser args: `args` is a argparse that should
         contain `init_method`, `rank` and `world_size`.
    :param Config cfg: `cfg`.

    """

    __master_path__ = None

    def __new__(cls):
        """Return a LocalMaster instance when run on local, else return a master instance."""
        mode = UserConfig().data.general.cluster_mode
        gpus = str(UserConfig().data.general.worker.gpus_per_job)
        if mode == ClusterMode.Single and gpus == "-1":
            return LocalMaster()
        else:
            return object.__new__(cls)

    def __init__(self):
        """Init master attrs, setup and start dask distributed cluster and local multiprocess pool."""
        self.cfg = copy.deepcopy(UserConfig().data.general)
        self.task_count = 0
        self.eval_count = self.cfg.worker.eval_count
        self.dask_env = DaskEnv(UserConfig().data.env,
                                self.__master_path__,
                                self.cfg.worker.gpus_per_job,
                                TaskOps(self.cfg).temp_path)
        status = self.dask_env.start()
        if not status or not self.dask_env.is_master:
            sys.exit(0)
        self._start_cluster()
        self._start_evaluator_multiprocess()
        self.t_queue = Queue()
        # now save GPU and Dloop Evaluator result.
        self.e_queue = utils.PairDictQueue()
        return

    def _start_cluster(self):
        """Set and start dask distributed cluster."""
        self.md = ClusterDaskDistributor(self.dask_env.master_address)
        self.client = self.md.get_client()
        local_host = None
        if "BATCH_CURRENT_HOST" in os.environ:
            local_host = os.environ["BATCH_CURRENT_HOST"]
        elif "BATCH_CUSTOM0_HOSTS" in os.environ:
            local_host = os.environ["BATCH_CUSTOM0_HOSTS"]
        plugin = WorkerEnv(self.dask_env.slave_proc_num,
                           self.dask_env.slave_gpus_per_proc,
                           local_host,
                           os.getpid(),
                           TaskOps(self.cfg).temp_path)
        self.client.register_worker_plugin(plugin)
        return

    def _start_evaluator_multiprocess(self):
        """Set and start local multiprocess pool."""
        self.dmd = LocalDistributor(self.eval_count)
        return

    @property
    def has_free_worker(self):
        """Property: check is has free dask worker.

        :return: return Ture if has free dask worker, otherwise return False.
        :rtype: bool

        """
        if self.md.process_queue_full():
            return False
        else:
            return True

    def run(self, worker):
        """Run a distributed_worker on different cluster.

        :param worker: A serializable object (callable and has `__call__`
             function) which need to be distributed calculaton.
        :type worker: object that the class was inherited from DistributedWorker.

        """
        if worker is None:
            return
        if worker.worker_type == utils.WorkerTypes.EVALUATOR:
            for sub_worker in worker.sub_worker_list:
                self.run(sub_worker)
                self.e_queue.add_new("{}::{}".format(sub_worker.step_name, sub_worker.worker_id),
                                     sub_worker.worker_type.name)
        elif worker.worker_type == utils.WorkerTypes.HAVA_D_EVALUATOR:
            p_id = self.task_count
            if worker.step_name and worker.worker_id:
                logging.info("master run EVALUATE_DLOOP")
                p_id = "{0}::{1}::{2}".format(worker.worker_type.name,
                                              worker.step_name,
                                              worker.worker_id)
            self.dmd.distribute(pid=p_id, func=worker, kwargs={})
            return p_id
        else:
            finished = False
            while not finished:
                if not self.md.process_queue_full():
                    p_id = self.task_count
                    if worker.step_name is not None and worker.worker_id is not None:
                        p_id = "{0}::{1}::{2}".format(worker.worker_type.name,
                                                      worker.step_name,
                                                      worker.worker_id)
                    self.md.distribute(client=self.client, pid=p_id,
                                       func=worker, kwargs={})
                    self.task_count = self.task_count + 1
                    return p_id
                else:
                    time.sleep(0.1)
        return

    def join(self):
        """Wait all workers to finished."""
        self.md.join()
        self.dmd.join()
        return

    def update_status(self):
        """Update Master queue status."""
        t_pid, _ = self.md.result_queue_get()
        if t_pid is not None:
            pid_splited = t_pid.split("::")
            if len(pid_splited) >= 3:
                type = pid_splited[0]
                pid = "{0}::{1}".format(pid_splited[1], pid_splited[2])
                if type == utils.WorkerTypes.TRAINER.name:
                    self.t_queue.put(pid)
                else:
                    self.e_queue.put(item=pid, type=type)
        dloop_pid = self.dmd.process_result_get()
        if dloop_pid is not None:
            pid_splited = dloop_pid.split("::")
            if len(pid_splited) >= 3:
                type = pid_splited[0]
                pid = "{0}::{1}".format(pid_splited[1], pid_splited[2])
            self.e_queue.put(item=pid, type=type)
        return

    def get_result_from_worker(self):
        """Get a result from a finished worker in dask cluster.

        :return: the pid and result of a finished worker if there are finished
             worker in queue, otherwise return(None, None).
        :rtype: (pid, result) or (None, None)

        """
        if not self.md.result_queue_empty():
            pid, result = self.md.result_queue_get()
            return pid, result
        else:
            return None, None

    def pop_finished_worker(self, train_worker=True):
        """Pop a finished dask worker's info, if there are finished dask worker in queue.

        :return: the finished worker info, include step_name and worker_id.
            eg. {"step_name":"round1", "worker_id":1}
        :rtype: dict or None

        """
        self.update_status()
        pid = None
        if train_worker and not self.t_queue.empty():
            pid = self.t_queue.get()
        else:
            pid = self.e_queue.get()
        if pid is None:
            return None
        else:
            pid_splited = pid.split("::")
            if len(pid_splited) < 2:
                return None
            else:
                return {"step_name": pid_splited[0],
                        "worker_id": pid_splited[1]}

    def pop_finished_train_worker(self):
        """Pop a finished evaluator worker's info, if there are finished evaluate workers in pool.

        :return: the finished worker info, include step_name and worker_id.
            eg. {"step_name":"round1", "worker_id":1}
        :rtype: dict or None

        """
        return self.pop_finished_worker(train_worker=True)

    def pop_finished_evaluate_worker(self):
        """Pop a finished evaluator worker's info, if there are finished evaluate workers in pool.

        :return: the finished worker info, include step_name and worker_id.
            eg. {"step_name":"round1", "worker_id":1}
        :rtype: dict or None

        """
        return self.pop_finished_worker(train_worker=False)

    def pop_all_finished_train_worker(self):
        """Pop all finished train worker's info.

        :return: a finished worker info list.
        :rtype: list of dict

        """
        worker_info_list = []
        finished_train_worker_info = self.pop_finished_train_worker()
        while finished_train_worker_info is not None:
            worker_info_list.append(finished_train_worker_info)
            finished_train_worker_info = self.pop_finished_train_worker()
        return worker_info_list

    def pop_all_finished_evaluate_worker(self):
        """Pop all finished evaluator worker's info.

        :return: a finished worker info list.
        :rtype: list of dict

        """
        worker_info_list = []
        finished_evaluate_worker_info = self.pop_finished_evaluate_worker()
        while finished_evaluate_worker_info is not None:
            worker_info_list.append(finished_evaluate_worker_info)
            finished_evaluate_worker_info = self.pop_finished_evaluate_worker()
        return worker_info_list

    def close_client(self):
        """Close cluster client."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
            del self.client

    @staticmethod
    def shutdown():
        """Shutdown all distributed cluster."""
        mode = UserConfig().data.general.cluster_mode
        gpus = str(UserConfig().data.general.worker.gpus_per_job)
        if mode == ClusterMode.Single and gpus == "-1":
            return
        try:
            logging.info("Try to shutdown cluster.")
            from vega.core.trainer.utils import get_write_ip_master_local
            from distributed import Client
            ip, port = get_write_ip_master_local()
            if ip is None or port is None:
                logging.info("Stand-alone mode, no need to shut down the cluster.")
                return
            shutdown_client = Client("{}:{}".format(ip, port))
            logging.info("Cluster will be shut down.")
            shutdown_client.shutdown()
            shutdown_client.close()
            del shutdown_client
            logging.info("Cluster is shut down.")
        except Exception as e:
            logging.error("Pipeline's cluster shutdown error: {}".format(str(e)))
            logging.error(traceback.format_exc())
