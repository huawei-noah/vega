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
import logging
import time
import threading
import uuid
import glob
from threading import Lock
from queue import Queue
from zeus.trainer import utils
from .distribution import ClusterDaskDistributor
from zeus.common import TaskOps
from zeus.common.general import General
from .worker_env import WorkerEnv
from .dask_env import DaskEnv
from zeus.trainer.deserialize import pickle_worker
from zeus.trainer.run_remote_worker import run_remote_worker
from zeus.report import ReportServer


class Master(object):
    """The Master Class is to create, maintain distribute framework and distribute calculate tasks.

    :param argparse.ArgumentParser args: `args` is a argparse that should
         contain `init_method`, `rank` and `world_size`.
    :param Config cfg: `cfg`.

    """

    __master_path__ = None

    def __init__(self, update_func=None):
        """Init master attrs, setup and start dask distributed cluster and local multiprocess pool."""
        self.cfg = General()
        self.task_count = 0
        self.eval_count = General.worker.eval_count
        self.dask_env = DaskEnv(General.env,
                                self.__master_path__,
                                General.devices_per_trainer,
                                TaskOps().temp_path)
        status = self.dask_env.start()
        if not status or not self.dask_env.is_master:
            sys.exit(0)
        self._start_cluster()
        self.t_queue = Queue()
        self.update_func = update_func
        self.evaluator_list = {}
        self._thread_runing = True
        self._lock = Lock()
        self._thread = self._run_monitor_thread()
        ReportServer().renew()
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
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ["ORIGIN_CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        self._remove_worker_number_file()
        plugin = WorkerEnv(self.dask_env.slave_proc_num,
                           self.dask_env.slave_device_num_per_proc,
                           local_host,
                           os.getpid(),
                           TaskOps().temp_path)
        self.client.register_worker_plugin(plugin)
        if "ORIGIN_CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["ORIGIN_CUDA_VISIBLE_DEVICES"]
        if "CUDA_VISIBLE_DEVICES" in os.environ and "ORIGIN_CUDA_VISIBLE_DEVICES" not in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        return

    def _remove_worker_number_file(self):
        _worker_number_file = os.path.join(TaskOps().temp_path, ".*worker_number")
        files = glob.glob(_worker_number_file)
        for _file in files:
            os.remove(_file)

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

    def run(self, worker, evaluator=None):
        """Run a distributed_worker on different cluster.

        :param worker: A serializable object (callable and has `__call__`
             function) which need to be distributed calculaton.
        :type worker: object that the class was inherited from DistributedWorker.

        """
        if worker is None:
            return

        if evaluator:
            with self._lock:
                self.evaluator_list[str(worker.worker_id)] = evaluator

        if worker.worker_type == utils.WorkerTypes.EVALUATOR:
            for sub_worker in worker.sub_worker_list:
                self.run(sub_worker)
        else:
            finished = False
            while not finished:
                if not self.md.process_queue_full():
                    p_id = self.task_count
                    if worker.step_name is not None and worker.worker_id is not None:
                        p_id = "{0}::{1}::{2}".format(worker.worker_type.name,
                                                      worker.step_name,
                                                      worker.worker_id)
                    pickle_id = uuid.uuid1().hex[:8]
                    pickle_worker(worker, pickle_id)
                    self.md.distribute(
                        client=self.client,
                        pid=p_id,
                        func=run_remote_worker,
                        kwargs={
                            "worker_id": worker.worker_id,
                            "worker_path": worker.get_local_worker_path(),
                            "id": pickle_id})
                    self.task_count = self.task_count + 1
                    return p_id
                else:
                    time.sleep(0.1)
        return

    @staticmethod
    def _monitor_thread(master):
        evaluator_counter = {}
        while master and master._thread_runing:
            worker_info_list = master._pop_all_finished_worker()
            if worker_info_list:
                for worker_info in worker_info_list:
                    worker_id = worker_info["worker_id"]
                    if str(worker_id) in master.evaluator_list.keys():
                        with master._lock:
                            evalutor = master.evaluator_list.pop(str(worker_id))
                        evaluator_counter[worker_id] = len(evalutor.sub_worker_list) - 1
                        master.run(evalutor)
                    elif master.update_func:
                        if worker_id not in evaluator_counter or evaluator_counter[worker_id] == 0:
                            master._update(worker_info["step_name"], worker_id)
                        else:
                            evaluator_counter[worker_id] -= 1
            time.sleep(0.1)

    def _update(self, step_name, worker_id):
        # Waiting report thread update all record
        time.sleep(0.5)
        if not self.update_func:
            return
        if self.update_func.__code__.co_varnames.index("step_name") == 1:
            self.update_func(step_name, worker_id)
        else:
            self.update_func({"step_name": step_name, "worker_id": worker_id})

    def _run_monitor_thread(self):
        try:
            logging.debug("Start master monitor thread.")
            monitor_thread = threading.Thread(target=Master._monitor_thread, args=(self,))
            monitor_thread.daemon = True
            monitor_thread.start()
            return monitor_thread
        except Exception as e:
            logging.error("Failed to run monitor thread.")
            raise e

    def join(self):
        """Wait all workers to finished."""
        self.md.join()
        return

    def update_status(self):
        """Update Master queue status."""
        t_pid, _ = self.md.result_queue_get()
        if t_pid is not None:
            pid_splited = t_pid.split("::")
            if len(pid_splited) >= 3:
                (_type, step_name, worker_id) = pid_splited
                pid = "{0}::{1}".format(step_name, worker_id)
                self.t_queue.put(pid)
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

    def _pop_finished_worker(self):
        """Pop a finished dask worker's info, if there are finished dask worker in queue.

        :return: the finished worker info, include step_name and worker_id.
            eg. {"step_name":"round1", "worker_id":1}
        :rtype: dict or None

        """
        self.update_status()
        pid = None
        if not self.t_queue.empty():
            pid = self.t_queue.get()
        if pid is None:
            return None
        else:
            pid_splited = pid.split("::")
            if len(pid_splited) < 2:
                return None
            else:
                return {"step_name": pid_splited[0],
                        "worker_id": pid_splited[1]}

    def _pop_all_finished_worker(self):
        """Pop all finished train worker's info.

        :return: a finished worker info list.
        :rtype: list of dict

        """
        worker_info_list = []
        finished_train_worker_info = self._pop_finished_worker()
        while finished_train_worker_info is not None:
            worker_info_list.append(finished_train_worker_info)
            finished_train_worker_info = self._pop_finished_worker()
        return worker_info_list

    def close_client(self):
        """Close cluster client."""
        ReportServer.stop()
        self._thread_runing = False
        # Waiting thread exit.
        time.sleep(1)
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
            del self.client
        # Waiting cluster closed
        time.sleep(1)

    def shutdown(self):
        """Close cluster client."""
        ReportServer.stop()
        self._thread_runing = False
        if self._thread:
            self._thread.join()
        # Waiting thread exit.
        time.sleep(1)
        if hasattr(self, "client") and self.client is not None:
            self.client.shutdown()
            self.client.close()
            del self.client
        # Waiting cluster closed
        time.sleep(1)
