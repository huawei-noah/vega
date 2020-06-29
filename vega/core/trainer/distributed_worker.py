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
import inspect
import os
import copy
import pickle
import subprocess
import logging
import traceback
from vega.core.common.task_ops import TaskOps
from .utils import kill_proc_tree
from vega.core.common.user_config import UserConfig
from vega.core.common.class_factory import ClassFactory


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

    def __init__(self, args=None):
        """Init DistributedWorker."""
        self.cfg = copy.deepcopy(args)
        super(DistributedWorker, self).__init__(self.cfg)
        # privates
        DistributedWorker.__worker_id__ = DistributedWorker.__worker_id__ + 1
        self._worker_id = DistributedWorker.__worker_id__
        # publics
        self.rank = 0
        self.world_size = 1
        self.worker_addr = ""
        self.worker_nccl_port = 16666
        self.timeout = int(float(self.cfg.worker.timeout) * 60 * 60)
        self.__env_config__ = (copy.deepcopy(UserConfig().data),
                               copy.deepcopy(ClassFactory.__configs__),
                               copy.deepcopy(ClassFactory.__registry__))
        return

    # basic decorators
    @classmethod
    def register(cls, *args):
        """Register classmethod function.

        A register modifier to register user's function that init DistributedWorker.
        in order to set user's syspath in different nodes.
        """
        def decorator(fn):
            file_name = inspect.getfile(fn)
            file_path = os.path.abspath(file_name)
            module_path = os.path.dirname(file_path)
            cls.__worker_path__ = module_path
            cls.__worker_module__ = os.path.splitext(os.path.basename(file_name))[0]
            cls.__worker_name__ = fn.__name__
            return fn
        return decorator

    # basic properties and setters

    @property
    def step_name(self):
        """Property: step_name."""
        return self.cfg.step_name

    @step_name.setter
    def step_name(self, value):
        """Setter: set step_name with value.

        :param value: step name
        :type value: str
        """
        self.cfg.step_name = value

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
        env = os.environ.copy()
        sub_pid_list = []
        if 'CUDA_VISIBLE_DEVICES' in env:
            try:
                first_gpu_id = env['CUDA_VISIBLE_DEVICES'].split(",")[0]
                env['VEGA_WORKER_PORT'] = '{}'.format(self.worker_nccl_port + int(first_gpu_id))
            except Exception:
                env['VEGA_WORKER_PORT'] = '{}'.format(self.worker_nccl_port)
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = "{0}:{1}".format(env['PYTHONPATH'], self.__worker_path__)
        elif self.__worker_id__ is not None and self.__worker_path__ is not None:
            env['PYTHONPATH'] = self.__worker_path__
        sub_pid = self._subprocess(rank=0, world_size=self.world_size,
                                   env=env, is_backend=False)
        sub_pid_list.append(sub_pid)
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
        worker_path = self.get_local_worker_path()
        worker_file = os.path.join(worker_path, 'worker_file_{0}_{1}.pickle'.format(self.worker_id, rank))
        with open(worker_file, "wb") as f:
            pickle.dump(self, f)
        env['RANK'] = "{}".format(rank)
        env['WORLD_SIZE'] = "{}".format(world_size)
        cmd = "import pickle;f=open('{0}', 'rb');augment = pickle.load(f);".format(worker_file)
        cmd = cmd + "from vega.core.common.user_config import UserConfig;"
        cmd = cmd + "from vega.core.common.class_factory import ClassFactory;"
        cmd = cmd + "user_config_data,ClassFactory.__configs__,ClassFactory.__registry__=augment.__env_config__;"
        cmd = cmd + "UserConfig().load(user_config_data);"
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
