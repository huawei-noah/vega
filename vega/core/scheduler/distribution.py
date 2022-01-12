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


"""
Classes for distribution.

Distributor Base Class, Dask Distributor Class and local Evaluator Distributor
Class. Distributor Classes are used in Master to init and maintain the cluster.
"""

import time
import multiprocessing
from threading import Lock


class DistributorBaseClass:
    """A base class for distributors."""

    def __init__(self):
        """Construct the DistributorBaseClass class."""
        raise NotImplementedError

    def distribute(self, pid, func, kwargs):
        """Distribute running a function in cluster, Abstract base function.

        :param pid: Unique `pid` for this func task.
        :type pid: str or int
        :param func: A serializable function or object(callable and has
            `__call__` function) which need to be distributed calculaton.
        :type func: function or object
        :param dict kwargs: Parameter of `func`.

        """
        raise NotImplementedError

    def close(self):
        """Clean the DistributorBaseClass after use, Abstract base function.

        e.g. close the connection to a DaskScheduler
        """
        pass


class ClusterDaskDistributor(DistributorBaseClass):
    """Distributor using a dask cluster.

    meaning that the calculation is spread over a cluster.

    :param str address: The `address` of dask-scheduler.
        eg. `tcp://127.0.0.1:8786`.

    """

    def __init__(self, address):
        """Set up a distributor that connects to a dask-scheduler to distribute the calculaton.

        :param address: the ip address and port number of the dask-scheduler.
        :type address: str
        """
        self.address = address
        self.future_set = set()
        self._queue_lock = Lock()

    def get_client(self):
        """Initialize a Client by pointing it to the address of a dask-scheduler.

        also, will init the worker count `self.n_workers` and two queue :
        `self.process_queue` and `self.result_queue` to save running process
        and results respectively.

        :return: return new client that is the primary entry point for users of
             dask.distributed.
        :rtype: distributed.Cient

        """
        from .run_dask import get_client
        from dask.distributed import Queue
        client = get_client(address=self.address)
        self.n_workers = len(client.scheduler_info()["workers"])
        self.process_queue = Queue(client=client, maxsize=self.n_workers)
        self.result_queue = Queue(client=client)
        return client

    def get_worker_count(self):
        """Get the worker count of current Client in dask-scheduler.

        :return: the worker count of current Client in dask-scheduler.
        :rtype: int

        """
        return self.n_workers

    def update_queues(self):
        """Update current client status, include all queue and set."""
        with self._queue_lock:
            finished_set = set()
            for f in self.future_set:
                pid = f[0]
                future = f[1]
                if future.done():
                    self.result_queue.put((pid, future.result()))
                    self.process_queue.get()
                    finished_set.add(f)
            for f in finished_set:
                self.future_set.remove(f)

    def result_queue_empty(self):
        """Update current client status, and return if the result queue is empty.

        :return: if the result queue is empty.
        :rtype: bool

        """
        self.update_queues()
        return self.result_queue.qsize() == 0

    def result_queue_get(self):
        """Get a (pid, reslut) pair from result queue if it is not empty.

        :return: first (pid, result) pair in result queue.
        :rtype: (str or int or None, a user-defined result or None)

        """
        self.update_queues()
        if self.result_queue.qsize() != 0:
            pid, result = self.result_queue.get()
            return pid, result
        else:
            return None, None

    def process_queue_full(self):
        """Check if current process queue is full.

        :return: if current process queue is full return True, otherwise False.
        :rtype: bool

        """
        self.update_queues()
        return self.process_queue.qsize() == self.n_workers

    def process_queue_empty(self):
        """Check if current process queue is empty.

        :return: if current process queue is empty return True, otherwise False.
        :rtype: bool

        """
        self.update_queues()
        return self.process_queue.qsize() == 0

    def distribute(self, client, pid, func, kwargs):
        """Submit a calculation task to cluster.

        the calculation task will be
        executed asynchronously on one worker of the cluster. the `client` is
        the cluster entry point, `pid` is a user-defined unique id for this
        task, `func` is the function or object that do the calculation,
        `kwargs` is the parameters for `func`.

        :param distributed.Client client: the target `client` to run this task.
        :param pid: unique `pid` to descript this task.
        :type pid: str or int(defined by user).
        :param func: A serializable function or object(callable and has
            `__call__` function) which need to be distributed calculaton.
        :type func: function or object.
        :param dict kwargs: Parameter of `func`.

        """
        with self._queue_lock:
            future = client.submit(func, **kwargs)
            f = (pid, future)
            self.future_set.add(f)
            self.process_queue.put(pid)

    def close(self, client):
        """Close the connection to the local Dask Scheduler.

        :param distributed.Client client: the target `client` to close.

        """
        client.close()

    def join(self):
        """Wait all process in process_queue to finish."""
        while not self.process_queue_empty():
            time.sleep(0.1)
        time.sleep(5)
        while not self.process_queue_empty():
            time.sleep(0.1)
        return


class LocalDistributor(DistributorBaseClass):
    """Distributor using a local multiprocessing pool.

    meaning that the calculation of function could spread to different
    process in pool.

    :param int n_workers: The size of process pool.

    """

    def __init__(self, n_workers):
        """Init the EvaluatorDistributor set n_worker.

        and init a process pool with size equal to n_worker.

        """
        self.n_workers = n_workers
        self.process_pool = multiprocessing.Pool(processes=n_workers)
        self.process_list = []

    def distribute(self, pid, func, kwargs):
        """Submit a calculation task to a localprocess pool.

        the calculation
        task will be executed asynchronously on one local process. the `pid`
        is a user-defined unique id for this task, `func` is the function or
        object that do the calculation, `kwargs` is the parameters for `func`.

        :param pid: unique `pid` to descript this task.
        :type pid: str or int(defined by user).
        :param func: A serializable function or object(callable and has
            `__call__` function) which need to be distributed calculaton.
        :type func: function or object.
        :param dict kwargs: Parameter of `func`.

        """
        res = self.process_pool.apply_async(func, **kwargs)
        self.process_list.append((pid, res))

    def process_result_get(self):
        """Update current process pool status.

        and return the pid of a first
        finished process in process list, and remove this process from process
        list, if there are any finished process, otherwise return
        None.

        :return: pid of a finished process.
        :rtype: pid or None.

        """
        if len(self.process_list) == 0:
            return None
        for p in self.process_list:
            pid = p[0]
            res = p[1]
            if res.ready():
                self.process_list.remove(p)
                return pid
        return None

    def close(self):
        """Close the current process pool."""
        self.process_pool.close()

    def join(self):
        """Wait all process in pool to finish."""
        for pid, res in self.process_list:
            if res is not None and not res.ready():
                res.wait()
        return
