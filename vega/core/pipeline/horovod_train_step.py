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

"""Horovod fully train."""

import os
import logging
import subprocess
from vega.common.general import General
from vega.common.class_factory import ClassFactory, ClassType
from vega.common import Status
from vega.report import ReportServer
from vega.core.pipeline.conf import PipeStepConfig
from vega.trainer.conf import TrainerConfig
from vega.common import FileOps
from .train_pipe_step import TrainPipeStep

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class HorovodTrainStep(TrainPipeStep):
    """TrainPipeStep is the implementation class of PipeStep.

    Fully train is the last pipe step in pipeline, we provide horovrd or local trainer
    for user to choose.
    """

    def do(self):
        """Start to run fully train with horovod or local trainer."""
        logger.info("HorovodTrainStep started.")
        General.cluster.horovod = True
        records = self._get_current_step_records()
        logger.debug("load pipestep records: {}".format(records))
        self._set_cluster_info()
        self.num_models = len(records)
        self.num_epochs = self.num_models * TrainerConfig.epochs
        self.update_status(Status.running)
        self._train_multi_models(records)
        ReportServer().output_step_all_records(step_name=self.task.step_name)
        ReportServer().backup_output_path()
        self.update_status(Status.finished)

    def _set_cluster_info(self):
        General.cluster.num_workers_per_node = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        General.cluster.num_nodes = max(General.cluster.num_nodes, 1 + len(General.cluster.slaves))
        General.cluster.num_workers = General.cluster.num_workers_per_node * General.cluster.num_nodes

    def train_model(self, trainer):
        """Train horovod model."""
        pwd_dir = os.path.dirname(os.path.abspath(__file__))
        cf_file = os.path.join(self.task.temp_path, 'cf.pickle')
        cf_content = {'registry': ClassFactory.__registry__,
                      'general_config': General().to_dict(),
                      'pipe_step_config': PipeStepConfig().to_dict(),
                      'model_desc': trainer.model_desc,
                      'worker_id': trainer.worker_id}
        FileOps.dump_pickle(cf_content, cf_file)
        worker_ips = '127.0.0.1'
        if General.cluster.master_ip is not None and General.cluster.master_ip != '127.0.0.1':
            worker_ips = General.cluster.master_ip
            for ip in General.cluster.slaves:
                worker_ips = worker_ips + ',' + ip
        cmd = ['bash', f'{pwd_dir}/horovod/run_horovod_train.sh',
               str(General.cluster.num_workers), cf_file, worker_ips, General.python_command]
        proc = subprocess.Popen(cmd, env=os.environ)
        proc.wait()
