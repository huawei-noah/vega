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

import os
import json
import logging
from distributed.diagnostics.plugin import WorkerPlugin


class WorkerEnv(WorkerPlugin):
    """WorkerEnv for add plugin in each worker in dask cluster.

    :param int device_quota: device num for each worker to use.

    """

    def __init__(self, device_quota):
        """Init the WorkerEnv."""
        self.device_quota = device_quota
        self._save_master_env()
        return

    def _get_devices(self, index, quota, env):
        all = os.environ[env].replace("'", "").replace("\"", "").replace(" ", "").split(",")
        npus = ",".join([all[index * quota + i] for i in range(quota)])
        return npus

    def _save_master_env(self):
        self.master_env = {
            "PATH": os.environ.get("PATH", None),
            "PYTHONPATH": os.environ.get("PYTHONPATH", None),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", None),
            "PWD": os.environ.get("PWD", None),
            "RANK_TABLE_FILE": os.environ.get("RANK_TABLE_FILE", None),
            "MINDSPORE_HCCL_CONFIG_PATH": os.environ.get("RANK_TABLE_FILE", None),
            "DLS_TASK_NUMBER": os.environ.get("DLS_TASK_NUMBER", None),
            "NPU_VISIBLE_DEVICES": os.environ.get("NPU_VISIBLE_DEVICES", None),
            "ASCEND_OPP_PATH": os.environ.get("ASCEND_OPP_PATH", None),
            "DEVICE_CATEGORY": os.environ.get("DEVICE_CATEGORY", None),
            "BACKEND_TYPE": os.environ.get("BACKEND_TYPE", None),
            "LD_PRELOAD": os.environ.get("LD_PRELOAD", None),
            "DLS_JOB_ID": os.environ.get("DLS_JOB_ID", None),
            "vega_python_command": os.environ.get("vega_python_command", None),
            "vega_timeout": os.environ.get("vega_timeout", None),
            "vega_world_size": os.environ.get("WORLD_SIZE", None),
            "vega_workers_list": os.environ.get("vega_workers_list", None),
            "vega_pytorch_hccl_port": os.environ.get("vega_pytorch_hccl_port", None),
        }

    def _restore_worker_env(self):
        for key, value in self.master_env.items():
            if value is not None:
                os.environ[key] = value

    def _set_visible_devices(self, worker):
        """Set visible devices to each worker env."""
        if os.environ['DEVICE_CATEGORY'] == 'GPU':
            _index = self._get_device_index(worker)
            devices = self._get_devices(_index, self.device_quota, "CUDA_VISIBLE_DEVICES")
            os.environ['CUDA_VISIBLE_DEVICES'] = devices
        elif os.environ['DEVICE_CATEGORY'] == 'NPU':
            ip = worker.ip
            _index = self._get_device_index(worker)
            device_id = self._get_devices(_index, self.device_quota, "NPU_VISIBLE_DEVICES").split(",")[0]
            os.environ['DEVICE_ID'] = device_id
            os.environ['ASCEND_DEVICE_ID'] = device_id
            rank_table_file = os.environ.get("RANK_TABLE_FILE", None)
            if rank_table_file:
                self._set_rank_info(device_id, rank_table_file, ip)
        else:
            raise Exception('device category must be GPU or NPU.')

    def _set_rank_info(self, device_id, rank_table_file, ip):
        try:
            with open(rank_table_file, 'r') as f:
                rank_table_json = json.loads(f.read())

            server_list = rank_table_json["server_list"]
            ips = [x["server_id"] for x in server_list]
            if len(ips) == 1:
                # single-node
                devices = rank_table_json['server_list'][0]['device']
                rank_id = list(filter(lambda x: x["device_id"] == device_id, devices))[0]["rank_id"]
                rank_size = str(len(devices))
                if "vega_pytorch_hccl_port" in os.environ:
                    port = os.environ['vega_pytorch_hccl_port']
                    os.environ['MASTER_ADDR'] = rank_table_json['server_list'][0]['device'][0]['device_ip']
                    os.environ['MASTER_PORT'] = rank_table_json['server_list'][0].get('server_port', port)
            else:
                # multi-nodes
                if "DLS_TASK_INDEX" in os.environ or 'VC_TASK_INDEX' in os.environ:
                    index = int(os.getenv('DLS_TASK_INDEX', os.getenv('VC_TASK_INDEX')))
                    devices = server_list[index]["device"]
                    rank_id = list(filter(lambda x: x["device_id"] == device_id, devices))[0]["rank_id"]
                    rank_size = str(sum([len(x["device"]) for x in server_list]))
                else:
                    if ip not in ips:
                        raise Exception(f"Worker IP {ip} not in rank table file ({ips}, {rank_table_file}). ")
                    devices = list(filter(lambda x: x["server_id"] == ip, server_list))[0]["device"]
                    rank_id = list(filter(lambda x: x["device_id"] == device_id, devices))[0]["rank_id"]
                    rank_size = str(sum([len(x["device"]) for x in server_list]))
            os.environ['RANK_ID'] = rank_id
            os.environ['RANK_SIZE'] = rank_size
        except Exception as e:
            logging.warn(f"wrong rank table file: {rank_table_file}, message: {e}")

    def _get_device_index(self, worker):
        ports_list = json.loads(os.environ["vega_workers_list"])
        (ip, port) = worker.worker_address.replace("//", "").split(":")[1:]
        _index = ports_list[ip].index(port)
        return _index

    def setup(self, worker):
        """Call back function for worker setup.

        here to get worker's local host name, and set worker visible gpu ids in
        CUDA_VISIBLE_DEVICES.

        """
        self._restore_worker_env()
        self._set_visible_devices(worker)
        return

    def teardown(self, worker):
        """Call back function for worker teardown."""
        return
