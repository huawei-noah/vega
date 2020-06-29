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
import subprocess
import logging
import time
from datetime import datetime
from distributed import Client
from ..trainer import utils
import shutil


class DaskEnv(object):
    """DaskEnv class is used to install, set and start dask-distributed cluser.

    :param argparse.ArgumentParser args: `args` is a argparse that should
         contain `init_method`, `rank` and `world_size`.
    :param str master_path: a python path that need to add in SYSPATH before
        all environment been setup.
    :param int gpus_per_job: .
    :param float worker_portion: the portion of workers that must wait to
         till connected with dask-scheduler.

    """

    def __init__(self, args, master_path, gpus_per_job, temp_path, worker_portion=1.0):
        """Init DaskEnv and set basic attrs."""
        self.args = args
        self.worker_portion = worker_portion
        self.__master_path__ = master_path
        self.is_master = False
        self.client = None
        self.master_address = None
        self.world_size = None
        self.slave_num = None
        self.slave_proc_num = None
        self.slave_gpus_per_proc = None
        self._set_slave_num(gpus_per_job)
        self._cluster_pid = []
        self.slaves = []
        self.temp_path = temp_path
        if 'slaves' in self.args and self.args.slaves:
            if not isinstance(self.args.slaves, list):
                self.args.slaves = [self.args.slaves]
            self.slaves = self.args.slaves

    def _set_slave_num(self, gpus_per_job):
        """Set slave node number.

        :param int gpus_per_job: Description of parameter `gpus_per_job`.

        """
        self.slave_num = self.args.world_size
        self.slave_proc_num = 1
        try:
            import torch
            system_gpu_num = 0
            if torch.cuda.is_available():
                system_gpu_num = torch.cuda.device_count()
            if gpus_per_job > 0 and gpus_per_job <= system_gpu_num:
                self.slave_gpus_per_proc = gpus_per_job
                self.slave_proc_num = int(system_gpu_num // gpus_per_job)
            else:
                self.slave_gpus_per_proc = system_gpu_num
                self.slave_proc_num = 1
        except ImportError:
            pass
        self.world_size = self.slave_num * self.slave_proc_num
        return

    def _get_address(self):
        """Get the master ip address and check if current node is master node.

        :return: if current node is master node.
        :rtype: bool.

        """
        ip, port = utils.get_master_address(self.args)
        self.master_address = "{}:{}".format(ip, port)
        return self.args.rank == 0

    def start(self):
        """Init DaskEnv, start cluster and wait all workers to connected with master.

        :return: if cluster is started successfully.
        :rtype: bool

        """
        self._install_dask()
        self._start_dask()
        self.is_master = self._get_address()
        if self.is_master:
            status = self._wait_workers()
            if status == 0:
                return False
        logging.info("Dask Server Start Success!")
        return True

    def stop(self):
        """TODO, stop the current cluster."""
        return

    def _install_dask(self):
        """Install dask package if dask.distributed not installed."""
        utils.install_and_import_local('dask', 'dask[complete]')

    def _start_dask(self):
        """Set PYTHONPATH, and start dask-scheduler on master node.

        then wait and start dask-worker on all nodes.
        """
        the_ip, the_port = utils.get_master_address(self.args)
        logging.info("master ip and port: {}:{}".format(the_ip, the_port))
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = "{0}:{1}".format(os.environ['PYTHONPATH'],
                                                        self.__master_path__)
        elif self.__master_path__ is not None:
            os.environ['PYTHONPATH'] = self.__master_path__

        # set distributed configs
        # os.environ['DASK_DISTRIBUTED__CLIENT__HEARTBEAT'] = '10s'

        if self.args.rank == 0:
            # host = utils.get_local_address()
            utils.write_ip(the_ip, the_port, self.args)
            address = "--node-ip-address={}".format(the_ip)
            port = "--port={}".format(the_port)
            try:
                Client("{}:{}".format(the_ip, the_port))
                logging.info("Reusing previous cluster:{}:{}".format(the_ip, the_port))
                return
            except Exception:
                logging.info("Dask-scheduler not start. Start dask-scheduler in master {}".format(the_ip))
            scheduler_p = subprocess.Popen(["dask-scheduler", port], env=os.environ)
            self._cluster_pid.append(scheduler_p.pid)
        time.sleep(10)
        self._check_dask_scheduler()
        master_host, master_port = utils.get_master_address(self.args)
        address = "tcp://{0}:{1}".format(master_host, master_port)
        logging.info("master host({}), address({}).".format(master_host, address))
        # nproc_set = "--nprocs={}".format(self.slave_proc_num)
        local_dir = "--local-directory={}/.vega_worker_{}".format(
            self.temp_path,
            datetime.now().strftime('%m%d.%H%M%S.%f')[:-3])
        # run dask-worker in master
        for i in range(self.slave_proc_num):
            worker_p = subprocess.Popen(["dask-worker", address, '--nthreads=1', '--nprocs=1',
                                        '--memory-limit=0', local_dir], env=os.environ)
            self._cluster_pid.append(worker_p.pid)
        # run dask-worker in each slaves.
        for slave_ip in self.slaves:
            for i in range(self.slave_proc_num):
                worker_p = subprocess.Popen(["ssh", slave_ip, shutil.which("dask-worker"), address, '--nthreads=1',
                                             '--nprocs=1', '--memory-limit=0', local_dir], env=os.environ)
                self._cluster_pid.append(worker_p.pid)

    def _check_dask_scheduler(self):
        """Check masker is start."""
        try:
            Client(self.master_address)
        except TimeoutError as ex:
            raise ex

    def _wait_workers(self):
        """Wait workers to connect with master, till worker_portion of workers are connected.

        :return: if worker_portion of workers are connected in 500s, return 1,
            otherwise return 0.
        :rtype: int

        """
        self.client = Client(self.master_address)
        logging.info("client scheduler info: {}".format(self.client.scheduler_info()))
        if int(self.world_size) <= 1:
            self.worker_portion = 1
        worker_count_min = int(self.world_size * self.worker_portion)

        for _ in range(100):
            time.sleep(5)
            n_workers = len(self.client.scheduler_info()["workers"])
            logging.info("Accessed Workers: {}".format(n_workers))
            if n_workers >= worker_count_min:
                workers = self.client.scheduler_info()["workers"]
                workers_list = []
                for k, _ in workers.items():
                    workers_list.append(k)
                logging.info("worker list: {}".format(workers_list))
                return 1
        return 0
