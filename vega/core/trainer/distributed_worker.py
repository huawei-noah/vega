# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Distributed worker for training and evaluating.

Distributed worker is the basic class of TrainWorker and EvaluatorWork,
it loads the pickle file into worker from master, and run the train_process
function of each distributed worker on local node, it also has the function
of timeout, killing the worker process which exceeds setting time.
"""
import os
import copy
import pickle
import subprocess
import logging
import traceback
import json
from vega.core.common.task_ops import TaskOps
from .utils import kill_proc_tree
from vega.core.common import UserConfig
from vega.core.common.class_factory import ClassFactory
from vega.search_space.networks import NetworkFactory
from vega.core.common.utils import switch_directory
from vega.core.common.general import General
from vega.core.common.config import obj2config


class DistributedWorker(TaskOps):
    """Class of Distributed Worker.

    This is a distributed worker used to load worker's pickle file,
    and run the process of training and evaluating.

    :param args: arguments from user config file
    :type args: dict or Config, default to None
    """

    # original params
    __worker_path__ = None
    __worker_module__ = None
    __worker_name__ = None
    # id params
    __worker_id__ = 0
    __config__ = None
    __general__ = None

    def __init__(self, args=None):
        """Init DistributedWorker."""
        super(DistributedWorker, self).__init__()
        # privates
        DistributedWorker.__worker_id__ = DistributedWorker.__worker_id__ + 1
        self._worker_id = DistributedWorker.__worker_id__
        # publics
        self.rank = 0
        self.world_size = 1
        self.worker_addr = ""
        self.worker_nccl_port = 16666
        self.timeout = int(float(General.worker.timeout) * 60 * 60)
        self.__env_config__ = (copy.deepcopy(UserConfig().data),
                               copy.deepcopy(ClassFactory.__configs__),
                               copy.deepcopy(ClassFactory.__registry__))
        self.__network_config__ = copy.deepcopy(NetworkFactory.__network_registry__)
        self.__general__ = obj2config(General)
        self.__worker_device_folder__ = os.path.join(self.temp_path, '.worker_device')
        if not os.path.exists(self.__worker_device_folder__):
            os.makedirs(self.__worker_device_folder__, exist_ok=True)
        return

    @property
    def worker_id(self):
        """Property: worker_id."""
        return self._worker_id

    @worker_id.setter
    def worker_id(self, value):
        """Setter: set worker_id with value.

        :param value: worker id
        :type value: int
        """
        self._worker_id = value

    def call_in_gpu(self):
        """Call function based on GPU devices."""
        env = os.environ.copy()
        sub_pid_list = []
        if 'CUDA_VISIBLE_DEVICES' in env:
            try:
                first_gpu_id = env['CUDA_VISIBLE_DEVICES'].split(",")[0]
                env['VEGA_WORKER_PORT'] = '{}'.format(self.worker_nccl_port + int(first_gpu_id))
            except Exception:
                env['VEGA_WORKER_PORT'] = '{}'.format(self.worker_nccl_port)
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = "{}:{}:{}".format(
                env['PYTHONPATH'], self.__worker_path__, os.path.abspath(os.curdir))
        elif self.__worker_id__ is not None and self.__worker_path__ is not None:
            env['PYTHONPATH'] = "{}:{}".format(
                self.__worker_path__, os.path.abspath(os.curdir))
        sub_pid = self._subprocess(rank=0, world_size=self.world_size,
                                   env=env, is_backend=False)
        sub_pid_list.append(sub_pid)
        return sub_pid_list

    def call_in_npu(self):
        """Call function based on NPU devices."""
        env = os.environ.copy()
        sub_pid_list = []
        npu_call_path = os.path.join(self.__worker_device_folder__, 'npu')
        if not os.path.exists(npu_call_path):
            os.makedirs(npu_call_path)
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = "{}:{}:{}".format(
                env['PYTHONPATH'], self.__worker_path__, os.path.abspath(os.curdir))
        elif self.__worker_id__ is not None and self.__worker_path__ is not None:
            env['PYTHONPATH'] = "{}:{}".format(
                self.__worker_path__, os.path.abspath(os.curdir))
        rank_file = env.get('RANK_TABLE_FILE')
        with open(rank_file, 'r') as f:
            rank_table_json = json.loads(f.read())
        if self.__env_config__[0].general.get('dft', False):
            env['RANK_SIZE'] = env['ORIGIN_RANK_SIZE']
            env['RANK_TABLE_FILE'] = env['ORIGIN_RANK_TABLE_FILE']
        else:
            env['RANK_SIZE'] = '1'
            env['DEVICE_ID'] = rank_table_json['group_list'][0]['instance_list'][0]['devices'][0]['device_id']
            env.pop('RANK_TABLE_FILE', None)
        with switch_directory(os.path.join(npu_call_path, 'device%s' % env['DEVICE_ID'])):
            sub_pid = self._subprocess(rank=0, world_size=1, env=env, is_backend=False)
        sub_pid_list.append(sub_pid)
        return sub_pid_list

    def __call__(self, *args, **kwargs):
        """Call function of distributed worker.

        To empty cuda memory, set environ,
        and do the subprocess function.

        :param *args: positional arguments
        :type *args: tuple
        :param ** kwargs: keyword argumnets
        :type ** kwargs: dict
        :return: 0
        """
        # empty the cuda memory first.
        # set Environment
        sub_pid_list = []
        if os.environ['DEVICE_CATEGORY'] == 'GPU':
            sub_pid_list = self.call_in_gpu()
        elif os.environ['DEVICE_CATEGORY'] == 'NPU':
            sub_pid_list = self.call_in_npu()
        # next we need to deal with the subprocess return status!!!
        logging.info("DistributedWorker finished!")
        for sub_pid in sub_pid_list:
            kill_proc_tree(pid=sub_pid)
        logging.info("DistributedWorker subprocess cleaned!")
        return 0

    def _subprocess(self, rank, world_size, env, is_backend=False):
        """Subprocess on each rank.

        Load pickle file into worker class, and use subprocess to run the
        train_process function.

        :param rank: node rank
        :type rank: int
        :param world_size: number of total nodes
        :type world_size: int
        :param env: environ
        :type env: dict
        :param is_backend: backend or not
        :type is_backend: bool
        """
        worker_path = self.get_local_worker_path(self.__general__.step_name, self.worker_id)
        worker_file = os.path.join(worker_path, 'worker_file_{0}_{1}.pickle'.format(self.worker_id, rank))
        with open(worker_file, "wb") as f:
            pickle.dump(self, f)
        env['RANK'] = "{}".format(rank)
        env['WORLD_SIZE'] = "{}".format(world_size)
        cmd = "import pickle;f=open('{0}', 'rb');augment = pickle.load(f);".format(worker_file)
        cmd = cmd + "from vega.core.common.user_config import UserConfig;"
        cmd = cmd + "from vega.core.common.class_factory import ClassFactory;"
        cmd = cmd + "from vega.search_space.networks import NetworkFactory;"
        cmd = cmd + "user_config_data,ClassFactory.__configs__,ClassFactory.__registry__=augment.__env_config__;"
        cmd = cmd + "NetworkFactory.__network_registry__=augment.__network_config__;"
        cmd = cmd + "UserConfig().load(user_config_data);"
        cmd = cmd + "from vega.core.common.loader import load_conf_from_desc;"
        cmd = cmd + "from vega.core.pipeline.conf import PipeStepConfig;"
        cmd = cmd + "load_conf_from_desc(PipeStepConfig, ClassFactory.__configs__);"
        cmd = cmd + "from vega.core.common.general import General;"
        cmd = cmd + "load_conf_from_desc(General, augment.__general__);"
        if 'VEGA_INIT_ENV' in os.environ:
            cmd = cmd + os.environ.copy()['VEGA_INIT_ENV']
        cmd = cmd + "augment.train_process()"
        if is_backend:
            proc = subprocess.Popen(['python3', '-c', cmd], close_fds=True, env=env)
            pid = proc.pid
        else:
            try:
                proc = subprocess.Popen(['python3', '-c', cmd], env=env)
                pid = proc.pid
                proc.wait(timeout=self.timeout)
            except Exception:
                logging.warn("Timeout worker has been killed.")
                logging.warn(traceback.print_exc())
        return pid

    def train_process(self):
        """Abstract base function for DistributedWorker to do the train process."""
        raise NotImplementedError
