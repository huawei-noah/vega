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
import json
import psutil
import subprocess
import traceback
import fcntl
from copy import deepcopy
from distributed.diagnostics.plugin import WorkerPlugin


class WorkerEnv(WorkerPlugin):
    """WorkerEnv for add plugin in each worker in dask cluster.

    :param int workers_each_node: worker count on each slave node.
    :param int device_quota: device num for each worker to use.
    :param str master_host_name: the dask cluster master host name.
    :param str master_pid: the process id of the master process.

    """

    def __init__(self, workers_each_node, device_quota, master_host_name, master_pid, temp_path):
        """Init the WorkerEnv."""
        self.workers_each_node = workers_each_node
        self.device_quota = device_quota
        self.master_host_name = master_host_name
        self.master_pid = master_pid
        self.device_list = []
        self._backend_type = os.environ["BACKEND_TYPE"]
        self.device_category = os.environ['DEVICE_CATEGORY']
        self._npu_visible_devices = os.environ.get('NPU_VISIBLE_DEVICES', None)
        self._npu_visible_devices = self._npu_visible_devices or os.environ.get('NPU-VISIBLE-DEVICES', None)
        self._batch_task_index = os.environ.get('BATCH_TASK_INDEX', None)
        self.temp_path = temp_path
        self.__worker_null_file__ = os.path.join(temp_path, '.vega_null')
        self.__worker_device_folder__ = os.path.join(temp_path, '.vega_device')
        self._cuda_devices = deepcopy(os.environ.get("ORIGIN_CUDA_VISIBLE_DEVICES", None))
        self._ori_rank_table_file = deepcopy(os.environ.get("ORIGIN_RANK_TABLE_FILE", None))
        if self._cuda_devices:
            _value = self._cuda_devices.replace("'", "").replace("\"", "").replace(" ", "").split(",")
            self._cuda_devices = [str(x) for x in _value]
        return

    def _init_worker_number_file(self, ip):
        """Use a local file to save a label to mark gpu id used for different workers on a same slave node."""
        _worker_number_file = os.path.join(self.temp_path, '.{}.worker_number'.format(ip))
        if not os.path.isfile(_worker_number_file):
            os.makedirs(os.path.dirname(_worker_number_file), exist_ok=True)
            fp = open(_worker_number_file, 'w')
            fcntl.flock(fp, fcntl.LOCK_EX)
            fp.write('{}'.format(0))
            fcntl.flock(fp, fcntl.LOCK_UN)
            fp.close()
        return _worker_number_file

    def _get_device_list(self, worker_number_file):
        """Get the cuda devices id list that are visible to current workers.

        :return: the current worker visible gpu id list.
        :rtype: list

        """
        current_count = 0
        with open(worker_number_file, 'r+') as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            f_str = fp.readline()
            try:
                # current_count = int(f_str.strip()) % self.workers_each_node
                current_count = int(f_str.strip())
            except Exception:
                pass
            with open(worker_number_file, 'w') as fn:
                fn.write('{}'.format(current_count + 1))
            fcntl.flock(fp, fcntl.LOCK_UN)
        device_list = []
        for i in range(current_count * self.device_quota, (current_count + 1) * self.device_quota):
            device_list.append('{}'.format(i))
        return device_list

    def _set_visible_devices(self):
        """Set visible devices to each worker env."""
        os.environ["BACKEND_TYPE"] = self._backend_type
        os.environ['DEVICE_CATEGORY'] = self.device_category
        if self.device_category == 'GPU':
            _device_list = self.device_list
            if self._cuda_devices:
                _device_list = [self._cuda_devices[int(i)] for i in _device_list]
            cuda_device_list_str = ",".join(_device_list)
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device_list_str
            # print("CUDA_VISIBLE_DEVICES: {}".format(cuda_device_list_str))
        elif self.device_category == 'NPU':
            self._fit_npu_device_list()
            origin_rank_file = self._ori_rank_table_file
            with open(origin_rank_file, 'r') as f:
                rank_table_json = json.loads(f.read())
            rank_table_json['server_count'] = 1
            group_info = rank_table_json['server_list']
            devices_info = []
            keep_idx = int(self._batch_task_index)
            instance_info = group_info[keep_idx]
            for device_id in self.device_list:
                device_id = int(device_id)
                devices_info.append(instance_info['device'][device_id])
            if len(devices_info) == 0:
                raise Exception('No matching devices info.')
            rank_table_json['server_list'] = [instance_info]
            rank_table_json['server_list'][0]['device'] = devices_info
            server_id = rank_table_json['server_list'][0]['server_id']
            new_rank_table_file = os.path.join(self.__worker_device_folder__,
                                               'rank_table_{0}_{1}.json'.format(server_id, self.device_list[0]))
            if not os.path.exists(self.__worker_device_folder__):
                os.makedirs(self.__worker_device_folder__, exist_ok=True)
            with open(new_rank_table_file, 'w') as f:
                f.write(json.dumps(rank_table_json))
            print('worker {} rank table json: {}'.format(self.device_list[0], rank_table_json))
            os.environ['RANK_TABLE_FILE'] = new_rank_table_file
            os.environ['RANK_SIZE'] = str(len(self.device_list))
            os.environ['DEVICE_ID'] = self.device_list[0]
            os.environ['ASCEND_DEVICE_ID'] = self.device_list[0]
            os.environ['MASTER_ADDR'] = rank_table_json['server_list'][0]['device'][0]['device_ip']
            os.environ['MASTER_PORT'] = rank_table_json['server_list'][0].get('server_port', '29688')
            os.environ['RANK_ID'] = rank_table_json['server_list'][0]['device'][0]['rank_id']
            # print("RANK_TABLE_FILE: {}".format(new_rank_table_file))
        else:
            raise Exception('device category must be GPU or NPU.')

    def _fit_npu_device_list(self):
        """Fit npu device list to actual visible devices."""
        visible_list = self._npu_visible_devices.split(',')
        new_device_list = list()
        for device_id in self.device_list:
            new_device_list.append(visible_list[int(device_id)])
        self.device_list = new_device_list

    def setup(self, worker):
        """Call back function for worker setup.

        here to get worker's local host name, and set worker visible gpu ids in
        CUDA_VISIBLE_DEVICES.

        """
        number_file = self._init_worker_number_file(worker.ip)
        self.device_list = self._get_device_list(number_file)
        self._set_visible_devices()
        return

    def teardown(self, worker):
        """Call back function for worker teardown."""
        return

    def transition_discard(self, key, start, finish, *args, **kwargs):
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
        print(" Plugin transition ")
        #
        if finish == 'ready' and len(self.device_list) > 0:
            try:
                current_pid = os.getpid()
                protect_pid_set = set()
                protect_pid_set.add(int(current_pid))
                # if self.master_host_name is not None and self.master_host_name == self.local_host_name:
                protect_pid_set.add(int(self.master_pid))
                try:
                    parent = psutil.Process(self.master_pid)
                    for p in parent.children(recursive=False):
                        protect_pid_set.add(int(p.pid))
                except Exception:
                    print("In slave node, master pid is not existed, process does not need to protect.")
                if self.device_category == 'GPU':
                    cuda_pid_set = set()
                    for id in self.device_list:
                        device = "/dev/nvidia{}".format(id)
                        fh = open(self.__worker_null_file__, "w")
                        p = subprocess.Popen(["fuser", "-v", device], stdout=subprocess.PIPE, stderr=fh)
                        p.wait()
                        fh.close()
                        sub_pids = p.stdout.read().split()
                        for spid in sub_pids[1:]:
                            cuda_pid_set.add(int(spid))
                    for spid in protect_pid_set:
                        if spid in cuda_pid_set:
                            cuda_pid_set.remove(spid)
                    # for spid in cuda_pid_set:
                    #     subprocess.call(["kill", "-9", "{}".format(spid)])
                    if cuda_pid_set:
                        print("Non-Vega process is using GPU, pids={}".format(cuda_pid_set))
            except Exception:
                print("Worker Plugin Error.")
                print(traceback.format_exc())
        print("cleaned the cuda memory...")
        return
