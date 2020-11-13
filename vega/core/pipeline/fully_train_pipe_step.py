# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Fully Train PipeStep that used in Pipeline."""
import os
import logging
import subprocess
import pickle
import vega
from .pipe_step import PipeStep
from zeus.common import ClassFactory, ClassType
from zeus.common import FileOps, TaskOps
from ..scheduler import create_master
from zeus.common.general import General
from zeus.report import Report, ReportRecord
from vega.core.pipeline.conf import PipeStepConfig, PipelineConfig

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class FullyTrainPipeStep(PipeStep):
    """FullyTrainPipeStep is the implementation class of PipeStep.

    Fully train is the last pipe step in pipeline, we provide horovrd or local trainer
    for user to choose.
    """

    def __init__(self):
        super().__init__()
        logger.info("init FullyTrainPipeStep...")

    def do(self):
        """Start to run fully train with horovod or local trainer."""
        logger.info("FullyTrainPipeStep started...")
        cls_trainer = ClassFactory.get_cls('trainer', "Trainer")
        if cls_trainer.config.distributed:
            self._do_distributed_fully_train()
        else:
            records = self._get_current_step_records()
            logger.debug("load pipestep records: {}".format(records))
            self.master = create_master(update_func=Report().update_report)
            self._train_multi_models(records)
            Report().output_step_all_records(
                step_name=self.task.step_name, weights_file=True, performance=True)
            self.master.close_client()
        Report().backup_output_path()

    def _get_current_step_records(self):
        step_name = self.task.step_name
        models_folder = PipeStepConfig.pipe_step.get("models_folder")
        records = []
        cur_index = PipelineConfig.steps.index(step_name)
        if cur_index >= 1 or models_folder:
            # records = Report().get_pareto_front_records(PipelineConfig.steps[cur_index - 1])
            if not models_folder:
                models_folder = FileOps.join_path(
                    TaskOps().local_output_path, PipelineConfig.steps[cur_index - 1])
            models_folder = models_folder.replace(
                "{local_base_path}", TaskOps().local_base_path)
            records = Report().load_records_from_model_folder(models_folder)
        else:
            records = [ReportRecord(step_name, 0)]
        logging.debug("Records: {}".format(records))
        for record in records:
            record.step_name = step_name
        return records

    def _train_single_model(self, model_desc=None, model_id=None):
        cls_trainer = ClassFactory.get_cls('trainer', "Trainer")
        step_name = self.task.step_name
        if model_desc is not None:
            sample = dict(worker_id=model_id, desc=model_desc, step_name=step_name)
            record = ReportRecord().load_dict(sample)
            logging.debug("Broadcast Record=%s", str(record))
            Report().broadcast(record)
            trainer = cls_trainer(model_desc=model_desc, id=model_id)
        else:
            trainer = cls_trainer(None, 0)
        # resume training
        if vega.is_torch_backend() and General._resume:
            trainer.load_checkpoint = True
            trainer._resume_training = True
        if cls_trainer.config.distributed:
            self._do_distributed_fully_train()
        else:
            self._do_single_fully_train(trainer)

    def _train_single_gpu_model(self, trainer):
        evaluator = self._get_evaluator(trainer.worker_id)
        self.master.run(trainer, evaluator)

    def _train_single_npu_model(self, trainer):
        temp_rank_file = os.environ.get('RANK_TABLE_FILE', None)
        temp_rank_size = os.environ['RANK_SIZE']
        os.environ.pop('RANK_TABLE_FILE', None)
        os.environ['RANK_SIZE'] = '1'
        evaluator = self._get_evaluator(trainer.worker_id)
        self.master.run(trainer, evaluator)
        if temp_rank_file is not None:
            os.environ['RANK_TABLE_FILE'] = temp_rank_file
        os.environ['RANK_SIZE'] = temp_rank_size

    def _do_single_fully_train(self, trainer):
        if os.environ['DEVICE_CATEGORY'] == 'GPU':
            self._train_single_gpu_model(trainer)
        elif os.environ['DEVICE_CATEGORY'] == 'NPU':
            self._train_single_npu_model(trainer)

    def _train_multi_models(self, records):
        for record in records:
            self._train_single_model(record.desc, record.worker_id)
        self.master.join()

    def _get_evaluator(self, worker_id):
        if not PipeStepConfig.evaluator_enable:
            return None
        cls_evaluator = ClassFactory.get_cls('evaluator', "Evaluator")
        evaluator = cls_evaluator({"step_name": self.task.step_name, "worker_id": worker_id})
        return evaluator

    def _do_horovod_fully_train(self):
        pwd_dir = os.path.dirname(os.path.abspath(__file__))
        cf_file = os.path.join(pwd_dir, 'cf.pickle')
        cf_content = {'registry': ClassFactory.__registry__,
                      'general_config': General().to_json(),
                      'pipe_step_config': PipeStepConfig().to_json()}
        with open(cf_file, 'wb') as f:
            pickle.dump(cf_content, f)
        cf_file_remote = os.path.join(self.task.local_base_path, 'cf.pickle')
        FileOps.copy_file(cf_file, cf_file_remote)
        if os.environ.get('DLS_TASK_NUMBER') is None:
            # local cluster
            worker_ips = '127.0.0.1'
            if General.cluster.master_ip is not None and General.cluster.master_ip != '127.0.0.1':
                worker_ips = General.cluster.master_ip
                for ip in General.cluster.slaves:
                    worker_ips = worker_ips + ',' + ip
            cmd = ['bash', '{}/horovod/run_cluster_horovod_train.sh'.format(pwd_dir),
                   str(self.world_device_size), cf_file_remote, worker_ips]
        else:
            # Roma
            cmd = ['bash', '{}/horovod/run_horovod_train.sh'.format(pwd_dir),
                   str(self.world_device_size), cf_file_remote]
        proc = subprocess.Popen(cmd, env=os.environ)
        proc.wait()

    def _do_hccl_fully_train(self):
        origin_parallel_fully_train = General.parallel_fully_train
        General.parallel_fully_train = True
        General.dft = True
        cls_trainer = ClassFactory.get_cls('trainer', "Trainer")
        self.master = create_master()
        workers_num = int(os.environ['RANK_SIZE'])
        for i in range(workers_num):
            trainer = cls_trainer(None, id=i)
            evaluator = self._get_evaluator(trainer.worker_id)
            self.master.run(trainer, evaluator)
        self.master.join()
        self.master.shutdown()
        General.parallel_fully_train = origin_parallel_fully_train
        General.dft = False

    def _do_distributed_fully_train(self):
        if os.environ['DEVICE_CATEGORY'] == 'GPU':
            self._do_horovod_fully_train()
        elif os.environ['DEVICE_CATEGORY'] == 'NPU':
            self._do_hccl_fully_train()

    @property
    def world_device_size(self):
        """World device size is world size * device count in each world."""
        import torch
        world_size = General.env.world_size
        device_nums = torch.cuda.device_count()
        num_devices = world_size * device_nums
        return num_devices
