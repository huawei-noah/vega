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
from vega.trainer import utils
from vega.common.file_ops import FileOps
from vega.common.general import General
import shutil


class DaskEnv(object):
    """DaskEnv class is used to install, set and start dask-distributed cluser.

    :param argparse.ArgumentParser args: `args` is a argparse that should
         contain `init_method`, `rank` and `world_size`.
    :param str master_path: a python path that need to add in SYSPATH before
        all environment been setup.
    :param int devices_per_trainer: .
    :param float worker_portion: the portion of workers that must wait to
         till connected with dask-scheduler.

    """

    def __init__(self, args, master_path, devices_per_trainer, temp_path, worker_portion=1.0):
        """Init DaskEnv and set basic attrs."""
        self.args = args
        self.worker_portion = worker_portion
        self.__master_path__ = master_path
        self.temp_path = temp_path
        self.is_master = False
        self.client = None
        self.master_address = None
        self.world_size = None
        self.slave_num = None
        self.slave_proc_num = None
        self.slave_device_num_per_proc = None
        self._set_slave_num(devices_per_trainer)
        self._cluster_pid = []
        self.slaves = []
        if 'slaves' in self.args and self.args.slaves:
            if not isinstance(self.args.slaves, list):
                self.args.slaves = [self.args.slaves]
            self.slaves = self.args.slaves

    def _set_slave_num(self, device_num):
        """Set slave node number.

        :param int device_num: Description of parameter `device_num`.

        """
        self.slave_num = self.args.world_size
        self.slave_proc_num = 1
        system_device_num = self._get_slave_device_num()
        if device_num > 0 and device_num <= system_device_num:
            self.slave_device_num_per_proc = device_num
            self.slave_proc_num = int(system_device_num // device_num)
        else:
            self.slave_device_num_per_proc = system_device_num
            self.slave_proc_num = 1
        self.world_size = self.slave_num * self.slave_proc_num
        if General.cluster.standalone_boot:
            self.world_size = General.cluster.num_workers
        return

    def _get_slave_device_num(self):
        device_category = os.environ['DEVICE_CATEGORY']
        system_device_num = 0
        if device_category == 'GPU':
            try:
                import torch
                system_device_num = 0
                if torch.cuda.is_available():
                    system_device_num = torch.cuda.device_count()
            except ImportError:
                pass
        elif device_category == 'NPU':
            try:
                system_device_num = len(os.environ['NPU-VISIBLE-DEVICES'].split(','))
            except Exception:
                pass
        else:
            raise Exception('device category must be GPU or NPU.')
        return system_device_num

    def start(self):
        """Init DaskEnv, start cluster and wait all workers to connected with master.

        :return: if cluster is started successfully.
        :rtype: bool

        """
        self._start_dask()
        self.is_master = self.args.rank == 0
        if self.is_master:
            status = self._wait_workers()
            if status == 0:
                return False
        logging.info("Dask Server Start Success!")
        return True

    def stop(self):
        """TODO, stop the current cluster."""
        return

    def _start_dask(self):
        """Set PYTHONPATH, and start dask-scheduler on master node.

        then wait and start dask-worker on all nodes.
        """
        the_ip, the_port = utils.get_master_address(self.args)
        logging.info("master ip and port: {}:{}".format(the_ip, the_port))
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = "{}:{}:{}".format(
                os.environ['PYTHONPATH'], self.__master_path__, os.path.abspath(os.curdir))
        elif self.__master_path__ is not None:
            os.environ['PYTHONPATH'] = "{}:{}".format(
                self.__master_path__, os.path.abspath(os.curdir))

        # set distributed configs
        # os.environ['DASK_DISTRIBUTED__CLIENT__HEARTBEAT'] = '10s'

        if self.args.rank == 0:
            # host = utils.get_local_address()
            utils.save_master_ip(the_ip, the_port, self.args)
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

        master_host, master_port = utils.get_master_address(self.args)
        address = "tcp://{0}:{1}".format(master_host, master_port)
        self.master_address = "{}:{}".format(master_host, master_port)
        logging.info("master host({}), address({}).".format(master_host, address))

        self._check_dask_scheduler()

        # nproc_set = "--nprocs={}".format(self.slave_proc_num)
        _local_dir = "{}/.vega_worker_{}".format(
            self.temp_path,
            datetime.now().strftime('%m%d.%H%M%S.%f')[:-3])
        FileOps.make_dir(_local_dir)
        local_dir = "--local-directory={}".format(_local_dir)
        # standalone boot mode, not dask-work is start by script
        if General.cluster.standalone_boot:
            return
        # run dask-worker in master
        for _ in range(self.slave_proc_num):
            worker_p = subprocess.Popen(["dask-worker", address, '--nthreads=1', '--nprocs=1',
                                         '--memory-limit=0', local_dir], env=os.environ)
            self._cluster_pid.append(worker_p.pid)
        # run dask-worker in each slaves.
        for slave_ip in self.slaves:
            for _ in range(self.slave_proc_num):
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
        logging.debug("client scheduler info: {}".format(self.client.scheduler_info()))
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
                slave_ips = list(set([item[6:].split(":")[0] for item in workers_list]))
                slave_ips.remove(General.cluster.master_ip)
                General.cluster.salves = slave_ips
                return 1
        return 0
