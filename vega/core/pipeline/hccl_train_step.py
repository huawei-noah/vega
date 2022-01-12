# -*- coding:utf-8 -*-

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

"""HCCL fully train."""

import os
import logging
import json
import vega
from vega.common.general import General
from vega.common.class_factory import ClassFactory, ClassType
from vega.common import Status, TaskOps
from vega.report import ReportServer
from vega.core.scheduler import create_master
from vega.trainer.conf import TrainerConfig
from vega.security.args import path_verify
from .train_pipe_step import TrainPipeStep

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class HcclTrainStep(TrainPipeStep):
    """TrainPipeStep is the implementation class of PipeStep.

    Fully train is the last pipe step in pipeline, we provide horovrd or local trainer
    for user to choose.
    """

    def do(self):
        """Start to run fully train with horovod or local trainer."""
        logger.info("HcclTrainStep started.")
        General.cluster.hccl = True
        records = self._get_current_step_records()
        logger.debug("load pipestep records: {}".format(records))
        self.num_models = len(records)
        self.num_epochs = self.num_models * TrainerConfig.epochs
        self.update_status(Status.running)
        self._set_nccl_ip_port()
        self._new_rank_table_file()
        self._set_ms_env()
        self._train_multi_models(records)
        ReportServer().output_step_all_records(step_name=self.task.step_name)
        ReportServer().backup_output_path()
        self.update_status(Status.finished)

    def train_model(self, trainer):
        """Train HCCL model."""
        origin_worker_id = trainer.worker_id

        General.parallel_fully_train = True
        General.devices_per_trainer = 1
        General._parallel = True

        self.master = create_master()
        for i in range(General.cluster.num_workers):
            worker_id = f"{origin_worker_id}-{i}" if i != 0 else origin_worker_id
            trainer.worker_id = worker_id
            trainer.hccl = True
            self.master.run(trainer)
        self.master.join()

        evaluator = self._get_evaluator(origin_worker_id)
        if evaluator:
            self.master.run(evaluator)
            self.master.join()

        self.master.close()

    def _set_nccl_ip_port(self):
        if not vega.is_torch_backend():
            return
        rank_file = os.getenv('RANK_TABLE_FILE', None)
        if not rank_file:
            raise ValueError('RANK_TABLE_FILE not in environ.')
        rank_file = os.path.realpath(rank_file)
        rank_file = path_verify(rank_file)
        with open(rank_file, 'r') as f:
            data = json.loads(f.read())
        General.cluster.hccl_server_ip = data['server_list'][0]['server_id']
        if "server_port" in data['server_list'][0]:
            General.cluster.hccl_port = int(data['server_list'][0]["server_port"])
        os.environ["vega_pytorch_hccl_port"] = str(General.cluster.hccl_port)
        logger.info(f"HCCL server: tcp://{General.cluster.hccl_server_ip}:{General.cluster.hccl_port}")

    def _new_rank_table_file(self):
        if not vega.is_torch_backend():
            return
        rank_file = os.getenv('RANK_TABLE_FILE', None)
        if not rank_file:
            raise ValueError('RANK_TABLE_FILE not in environ.')
        rank_file = os.path.realpath(rank_file)
        rank_file = path_verify(rank_file)
        with open(rank_file, 'r') as f:
            data = json.loads(f.read())
        device_ids = os.environ["NPU_VISIBLE_DEVICES"].split(",")
        changed = False
        num_server = len(data['server_list'])
        rank_size = 0
        rank_index = 0
        for server_id in range(num_server):
            origin_devices = data['server_list'][server_id]['device']
            if len(device_ids) != len(origin_devices):
                changed = True
            new_devices = []
            for device in origin_devices:
                if device["device_id"] in device_ids:
                    device["rank_id"] = str(rank_index)
                    rank_index += 1
                    new_devices.append(device)
            data['server_list'][server_id]['device'] = new_devices
            rank_size += len(new_devices)
        if changed:
            rank_file = os.path.join(TaskOps().temp_path, "rank_table_file.json")
            with open(rank_file, "w") as f:
                json.dump(data, f)
            os.environ["RANK_TABLE_FILE"] = rank_file
            os.environ["RANK_SIZE"] = str(rank_size)

    def _set_ms_env(self):
        if vega.is_ms_backend():
            os.environ["MINDSPORE_HCCL_CONFIG_PATH"] = os.environ["RANK_TABLE_FILE"]
