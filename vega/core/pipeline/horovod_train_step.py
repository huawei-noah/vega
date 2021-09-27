# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Horovod fully train."""

import os
import logging
import subprocess
import pickle
import vega
from .train_pipe_step import TrainPipeStep
from vega.common.general import General
from vega.common.class_factory import ClassFactory, ClassType
from vega.common import Status
from vega.report import ReportServer
from vega.core.pipeline.conf import PipeStepConfig
from vega.trainer.conf import TrainerConfig

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
        with open(cf_file, 'wb') as f:
            pickle.dump(cf_content, f)
        if os.environ.get('DLS_TASK_NUMBER') is None:
            # local cluster
            worker_ips = '127.0.0.1'
            if General.cluster.master_ip is not None and General.cluster.master_ip != '127.0.0.1':
                worker_ips = General.cluster.master_ip
                for ip in General.cluster.slaves:
                    worker_ips = worker_ips + ',' + ip
            cmd = ['bash', f'{pwd_dir}/horovod/run_horovod_train.sh',
                   str(General.cluster.num_workers), cf_file, worker_ips, General.python_command]
        else:
            # Roma
            cmd = ['bash', '/home/work/run_horovod_train.sh',
                   str(General.cluster.num_workers), cf_file]
        proc = subprocess.Popen(cmd, env=os.environ)
        proc.wait()
