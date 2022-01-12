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

"""The main part of the cluster framework.

The DaskEnv Class which used in Master to init and set basic dask-distributed
environment.
"""

import json
import os
import logging
import time
import uuid
from datetime import datetime
from vega.trainer import utils
from vega.common import FileOps
from vega.common.general import General
from vega.core.scheduler.run_dask import get_client, run_scheduler, run_local_worker, run_remote_worker, get_address


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
        General.cluster.num_workers = self.world_size
        General.cluster.num_nodes = self.slave_num
        General.cluster.num_workers_per_node = self.slave_proc_num
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
                system_device_num = len(os.environ['NPU_VISIBLE_DEVICES'].split(','))
            except Exception:
                logging.debug("Failed to get NPU_VISIBLE_DEVICES in environ.")
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
        """Stop the current cluster."""
        return

    def _start_dask(self):
        """Set PYTHONPATH, and start dask-scheduler on master node.

        then wait and start dask-worker on all nodes.
        """
        master_ip, master_port = utils.get_master_address(self.args)
        logging.info("master ip and port: {}:{}".format(master_ip, master_port))
        logging.info("Initializing cluster. Please wait.")
        if "PYTHONPATH" not in os.environ:
            os.environ["PYTHONPATH"] = ""
        if self.__master_path__ not in os.environ["PYTHONPATH"].split(":"):
            os.environ["PYTHONPATH"] += f":{self.__master_path__}"
        if os.path.abspath(os.curdir) not in os.environ["PYTHONPATH"].split(":"):
            os.environ["PYTHONPATH"] += f":{os.path.abspath(os.curdir)}"
        if self.args.rank == 0:
            try:
                get_client(get_address(master_ip, master_port))
                logging.info("Reusing previous cluster:{}:{}".format(master_ip, master_port))
                return
            except Exception:
                logging.info("Dask-scheduler not start. Start dask-scheduler in master {}".format(master_ip))
            scheduler_file = f"{self.temp_path}/.scheduler/scheduler.tmp"
            FileOps.make_base_dir(scheduler_file)
            scheduler_p = run_scheduler(ip=master_ip, port=master_port, tmp_file=scheduler_file)
            self._cluster_pid.append(scheduler_p.pid)

        self.master_address = get_address(master_ip, master_port)
        logging.info("master host({}), address({}).".format(master_ip, self.master_address))

        self._check_dask_scheduler()

        local_dir = f"{self.temp_path}/.vega_worker"
        FileOps.make_dir(local_dir)
        # standalone boot mode, not dask-work is start by script
        if General.cluster.standalone_boot:
            return
        # run dask-worker in master
        for _ in range(self.slave_proc_num):
            local_master_dir = local_dir + '/{}'.format(uuid.uuid1().hex[:8])
            FileOps.make_dir(local_master_dir)
            worker_p = run_local_worker(slave_ip=master_ip, address=self.master_address, local_dir=local_master_dir)
            self._cluster_pid.append(worker_p.pid)
        # run dask-worker in each slaves.
        for slave_ip in self.slaves:
            for _ in range(self.slave_proc_num):
                local_slaves_dir = local_dir + '/{}'.format(uuid.uuid1().hex[:8])
                FileOps.make_dir(local_slaves_dir)
                worker_p = run_remote_worker(slave_ip=slave_ip, address=self.master_address, local_dir=local_slaves_dir)
                self._cluster_pid.append(worker_p.pid)

    def _check_dask_scheduler(self):
        """Check masker is start."""
        try:
            get_client(self.master_address)
        except TimeoutError as ex:
            raise ex

    def _wait_workers(self):
        """Wait workers to connect with master, till worker_portion of workers are connected.

        :return: if worker_portion of workers are connected in 500s, return 1,
            otherwise return 0.
        :rtype: int

        """
        self.client = get_client(self.master_address)
        logging.debug("client scheduler info: {}".format(self.client.scheduler_info()))
        if int(self.world_size) <= 1:
            self.worker_portion = 1
        worker_count_min = int(self.world_size * self.worker_portion)

        for _ in range(100):
            time.sleep(1)
            n_workers = len(self.client.scheduler_info()["workers"])
            logging.info("Accessed Workers: {}".format(n_workers))
            if n_workers >= worker_count_min:
                workers = self.client.scheduler_info()["workers"]
                workers_list = []
                workers_port = {}
                for k, _ in workers.items():
                    workers_list.append(k)
                    (ip, port) = k.replace("//", "").split(":")[1:]
                    if ip in workers_port:
                        workers_port[ip].append(port)
                    else:
                        workers_port[ip] = [port]
                os.environ["vega_workers_list"] = json.dumps(workers_port)
                logging.info("worker list: {}".format(workers_list))
                slave_ips = list(set([item[6:].split(":")[0] for item in workers_list]))
                slave_ips.remove(General.cluster.master_ip)
                General.cluster.salves = slave_ips
                return 1
        return 0
