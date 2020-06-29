# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The main part of the cluster framework.

The DaskEnv Class which used in Master to init and set basic dask-distributed
environment.
"""
import os
import psutil
import subprocess
import traceback
import logging
import fcntl
from distributed.diagnostics.plugin import WorkerPlugin


class WorkerEnv(WorkerPlugin):
    """WorkerEnv for add plugin in each worker in dask cluster.

    :param int workers_each_node: worker count on each slave node.
    :param int gpu_quota: gpus_per_job for each worker to use.
    :param str master_host_name: the dask cluster master host name.
    :param str master_pid: the process id of the master process.

    """

    def __init__(self, workers_each_node, gpu_quota, master_host_name, master_pid, temp_path):
        """Init the WorkerEnv."""
        self.workers_each_node = workers_each_node
        self.gpu_quota = gpu_quota
        self.master_host_name = master_host_name
        self.local_host_name = None
        self.master_pid = master_pid
        self.device_list = []
        self.__worker_number_file__ = os.path.join(temp_path, '.vega_worker_env_gpu')
        self.__worker_null_file__ = os.path.join(temp_path, '.vega_null')
        return

    def _set_cuda_env(self):
        """Use a local file to save a label to mark gpu id used for differernt workers on a same slave node."""
        if not os.path.isfile(self.__worker_number_file__):
            fp = open(self.__worker_number_file__, 'w')
            fcntl.flock(fp, fcntl.LOCK_EX)
            fp.write('{}'.format(0))
            fcntl.flock(fp, fcntl.LOCK_UN)
            fp.close()
        return

    def _get_cuda_devices(self):
        """Get the cuda devices id list that are visible to current workers.

        :return: the current worker visible gpu id list.
        :rtype: list

        """
        current_count = 0
        with open(self.__worker_number_file__, 'r+') as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            f_str = fp.readline()
            try:
                current_count = int(f_str.strip()) % self.workers_each_node
            except Exception:
                pass
            with open(self.__worker_number_file__, 'w') as fn:
                fn.write('{}'.format(current_count + 1))
            fcntl.flock(fp, fcntl.LOCK_UN)
        device_list = []
        for i in range(current_count * self.gpu_quota, (current_count + 1) * self.gpu_quota):
            device_list.append('{}'.format(i))
        return device_list

    def setup(self, worker):
        """Call back function for worker setup.

        here to get worker's local host name, and set worker visible gpu ids in
        CUDA_VISIBLE_DEVICES.

        """
        if "BATCH_CURRENT_HOST" in os.environ:
            self.local_host_name = os.environ["BATCH_CURRENT_HOST"]
        elif "BATCH_CUSTOM0_HOSTS" in os.environ:
            self.local_host_name = os.environ["BATCH_CUSTOM0_HOSTS"]
        self._set_cuda_env()
        self.device_list = self._get_cuda_devices()
        cuda_device_list_str = ",".join(self.device_list)
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device_list_str
        logging.info("CUDA_VISIBLE_DEVICES:" + cuda_device_list_str)
        return

    def teardown(self, worker):
        """Call back function for worker teardown."""
        return

    def transition(self, key, start, finish, *args, **kwargs):
        """Call back function for worker status transition.

        here to clean the gpu memory whe worker status turn to `ready`,
        use `fuser -v` list all pid that use cuda, and filter the master's
        processes, and kill all other processes.

        :param str key: Description of parameter `key`.
        :param str start: Start state of the transition.
            One of waiting, ready, executing, long-running, memory, error.
        :param str finish: Final state of the transition.
        :param type * args: Description of parameter `*args`.
        :param type ** kwargs: Description of parameter `**kwargs`.

        """
        logging.info(" Plugin transition ")
        #
        if finish == 'ready' and len(self.device_list) > 0:
            try:
                current_pid = os.getpid()
                protect_pid_set = set()
                protect_pid_set.add(int(current_pid))
                cuda_pid_set = set()
                # if self.master_host_name is not None and self.master_host_name == self.local_host_name:
                protect_pid_set.add(int(self.master_pid))
                try:
                    parent = psutil.Process(self.master_pid)
                    for p in parent.children(recursive=False):
                        protect_pid_set.add(int(p.pid))
                except Exception:
                    logging.debug("In slave node, master pid is not existed, process does not need to protect.")
                for id in self.device_list:
                    device = "/dev/nvidia{}".format(id)
                    fh = open(self.__worker_null_file__, "w")
                    p = subprocess.Popen(["fuser", "-v", device], stdout=subprocess.PIPE, stderr=fh)
                    p.wait()
                    fh.close()
                    sub_pids = p.stdout.read().split()
                    for spid in sub_pids:
                        cuda_pid_set.add(int(spid))
                for spid in protect_pid_set:
                    if spid in cuda_pid_set:
                        cuda_pid_set.remove(spid)
                if cuda_pid_set:
                    logging.info("Non-Vega process is using GPU, pids={}".format(cuda_pid_set))
                # for spid in cuda_pid_set:
                #     subprocess.call(["kill", "-9", "{}".format(spid)])
            except Exception:
                logging.error("Worker Plugin Error.")
                logging.error(traceback.format_exc())
        logging.info("cleaned the cuda memory...")
        return
