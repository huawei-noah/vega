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
from vega.common.general import General
from vega.common.class_factory import ClassFactory, ClassType
from vega.common import FileOps, TaskOps, Status
from vega.report import ReportServer, ReportRecord, ReportClient
from vega.core.scheduler import create_master
from vega.core.pipeline.conf import PipeStepConfig, PipelineConfig
from vega.trainer.conf import TrainerConfig

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class TrainPipeStep(PipeStep):
    """TrainPipeStep is the implementation class of PipeStep.

    Fully train is the last pipe step in pipeline, we provide horovrd or local trainer
    for user to choose.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distributed_training = not General._parallel and TrainerConfig.distributed
        logger.info("init TrainPipeStep...")

    def do(self):
        """Start to run fully train with horovod or local trainer."""
        super().do()
        logger.info("TrainPipeStep started...")
        records = self._get_current_step_records()
        logger.debug("load pipestep records: {}".format(records))
        self.num_models = len(records)
        self.num_epochs = self.num_models * TrainerConfig.epochs
        self.update_status(Status.running)
        self.master = create_master()
        self._train_multi_models(records)
        self.master.join()
        ReportServer().output_step_all_records(step_name=self.task.step_name)
        self.master.close()
        ReportServer().backup_output_path()
        self.update_status(Status.finished)

    def _get_current_step_records(self):
        step_name = self.task.step_name
        models_folder = PipeStepConfig.pipe_step.get("models_folder")
        models_folder = models_folder or PipeStepConfig.pipe_step.get("hps_folder")
        cur_index = PipelineConfig.steps.index(step_name)
        if cur_index >= 1 or models_folder:
            if not models_folder:
                models_folder = FileOps.join_path(
                    TaskOps().local_output_path, PipelineConfig.steps[cur_index - 1])
            models_folder = models_folder.replace(
                "{local_base_path}", TaskOps().local_base_path)
            records = ReportServer().load_records_from_model_folder(models_folder)
        else:
            records = [ReportRecord(step_name, 0)]
        logging.debug("Records: {}".format(records))
        for record in records:
            record.step_name = step_name
        return records

    def _train_single_model(self, model_desc=None, hps=None, model_id=None, weights_file=None):
        cls_trainer = ClassFactory.get_cls(ClassType.TRAINER, PipeStepConfig.trainer.type)
        step_name = self.task.step_name
        if model_desc is not None:
            sample = dict(worker_id=model_id, desc=model_desc, step_name=step_name)
            record = ReportRecord().load_dict(sample)
            logging.debug("update record=%s", str(record))
            trainer = cls_trainer(model_desc=model_desc, hps=hps, id=model_id, pretrained_model_file=weights_file)
        else:
            trainer = cls_trainer(None, 0, hps=hps)
            record = ReportRecord(trainer.step_name, trainer.worker_id, desc=trainer.model_desc, hps=hps)
        ReportClient().update(**record.to_dict())
        # resume training
        if vega.is_torch_backend() and General._resume:
            trainer.load_checkpoint = True
            trainer._resume_training = True
        if self._distributed_training:
            self._do_distributed_fully_train(trainer)
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
            weights_file = record.weights_file if PipeStepConfig.pipe_step.get("load_weights", True) else None
            self._train_single_model(
                model_desc=record.desc, hps=record.hps, model_id=record.worker_id, weights_file=weights_file)

    def _get_evaluator(self, worker_id):
        if not PipeStepConfig.evaluator_enable:
            return None
        cls_evaluator = ClassFactory.get_cls('evaluator', "Evaluator")
        evaluator = cls_evaluator({"step_name": self.task.step_name, "worker_id": worker_id})
        return evaluator

    def _do_horovod_fully_train(self, trainer):
        # records = self._get_current_step_records()
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
                   str(self.world_device_size), cf_file, worker_ips, General.python_command]
        else:
            # Roma
            cmd = ['bash', '/home/work/run_horovod_train.sh',
                   str(self.world_device_size), cf_file]
        proc = subprocess.Popen(cmd, env=os.environ)
        proc.wait()

    def _do_hccl_fully_train(self, trainer):
        origin_worker_id = trainer.worker_id
        model_desc = trainer.model_desc
        del trainer

        os.environ['RANK_SIZE'] = os.environ['ORIGIN_RANK_SIZE']
        os.environ['RANK_TABLE_FILE'] = os.environ['ORIGIN_RANK_TABLE_FILE']
        origin_parallel_fully_train = General.parallel_fully_train
        origin_parallel = General._parallel
        General.parallel_fully_train = True
        General.dft = True
        General._parallel = True

        cls_trainer = ClassFactory.get_cls(ClassType.TRAINER, PipeStepConfig.trainer.type)
        self.master = create_master()
        workers_num = int(os.environ['RANK_SIZE'])
        for i in range(workers_num):
            worker_id = "{}-{}".format(origin_worker_id, i)
            trainer = cls_trainer(model_desc, id=worker_id)
            evaluator = self._get_evaluator(worker_id) if os.environ['DEVICE_ID'] == "0" else None
            self.master.run(trainer, evaluator)

        self.master.join()
        self.master.close()
        General.parallel_fully_train = origin_parallel_fully_train
        General.dft = False
        General._parallel = origin_parallel

    def _do_distributed_fully_train(self, trainer):
        if os.environ['DEVICE_CATEGORY'] == 'GPU':
            self._do_horovod_fully_train(trainer)
        elif os.environ['DEVICE_CATEGORY'] == 'NPU':
            self._do_hccl_fully_train(trainer)

    @property
    def world_device_size(self):
        """World device size is world size * device count in each world."""
        import torch
        world_size = General.env.world_size
        device_nums = torch.cuda.device_count()
        num_devices = world_size * device_nums
        return num_devices
