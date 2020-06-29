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
import glob
import traceback
from copy import deepcopy
import numpy as np
from .pipe_step import PipeStep
from ..common.class_factory import ClassFactory, ClassType
from vega.core.common import UserConfig, FileOps, TaskOps, Config
from ..scheduler import Master, LocalMaster


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
        cls_trainer = ClassFactory.get_cls('trainer')
        trainer_cfg = ClassFactory.__configs__.get('trainer')
        setattr(trainer_cfg, 'save_best_model', True)
        if cls_trainer.cfg.get('horovod', False):
            self._do_horovod_fully_train()
        else:
            cfg = Config(deepcopy(UserConfig().data))
            step_name = cfg.general.step_name
            pipe_step_cfg = cfg[step_name].pipe_step
            if "esr_models_file" in pipe_step_cfg and pipe_step_cfg.esr_models_file is not None:
                self.master = Master()
                self._train_esr_models(pipe_step_cfg.esr_models_file)
            elif "models_folder" in pipe_step_cfg and pipe_step_cfg.models_folder is not None:
                self.master = Master()
                self._train_multi_models(pipe_step_cfg.models_folder)
            else:
                self.master = LocalMaster()
                self._train_single_model()
            self.master.join()
            self.master.close_client()
        self._backup_output_path()

    def _train_single_model(self, desc_file=None, model_id=None):
        cls_trainer = ClassFactory.get_cls('trainer')
        if desc_file is not None:
            cls_trainer.cfg.model_desc_file = desc_file
            model_cfg = ClassFactory.__configs__.get('model')
            if model_cfg:
                setattr(model_cfg, 'model_desc_file', desc_file)
            else:
                setattr(ClassFactory.__configs__, 'model', Config({'model_desc_file': desc_file}))
        if cls_trainer.cfg.get('horovod', False):
            self._do_horovod_fully_train()
        else:
            trainer = cls_trainer(None, id=model_id)
            self.master.run(trainer)

    def _train_multi_models(self, models_folder):
        models_folder = models_folder.replace("{local_base_path}", self.task.local_base_path)
        models_folder = os.path.abspath(models_folder)
        model_desc_files = glob.glob("{}/model_desc_*.json".format(models_folder))
        for desc_file in model_desc_files:
            id = os.path.splitext(os.path.basename(desc_file))[0][11:]
            logger.info("Begin train model, id={}, desc={}".format(id, desc_file))
            try:
                self._train_single_model(desc_file, id)
            except Exception as ex:
                logger.error("Failed to train model, id={}, desc={}".format(id, desc_file))
                logger.error("Traceback message:")
                logger.error(traceback.format_exc())

    def _train_esr_models(self, esr_models_file):
        esr_models_file = esr_models_file.replace("{local_base_path}", self.task.local_base_path)
        esr_models_file = os.path.abspath(esr_models_file)
        archs = np.load(esr_models_file)
        for i, arch in enumerate(archs):
            cls_trainer = ClassFactory.get_cls('trainer')
            cls_trainer.cfg.model_arch = arch
            model_cfg = ClassFactory.__configs__.get('model')
            if model_cfg:
                setattr(model_cfg, 'model_arch', arch)
            else:
                setattr(ClassFactory.__configs__, 'model', Config({'model_arch': arch}))
            if cls_trainer.cfg.get('horovod', False):
                self._do_horovod_fully_train()
            else:
                trainer = cls_trainer(None, id=i)
                self.master.run(trainer)

    def _do_horovod_fully_train(self):
        """Call horovod bash file to load pickle files saved by vega."""
        pwd_dir = os.path.dirname(os.path.abspath(__file__))
        cf_file = os.path.join(pwd_dir, 'cf.pickle')
        cf_content = {'configs': ClassFactory.__configs__,
                      'registry': ClassFactory.__registry__,
                      'data': UserConfig().__data__}
        with open(cf_file, 'wb') as f:
            pickle.dump(cf_content, f)
        cf_file_remote = os.path.join(self.task.local_base_path, 'cf.pickle')
        FileOps.copy_file(cf_file, cf_file_remote)
        if os.environ.get('DLS_TASK_NUMBER') is None:
            # local cluster
            worker_ips = '127.0.0.1'
            if UserConfig().data.general.cluster.master_ip is not None and \
               UserConfig().data.general.cluster.master_ip != '127.0.0.1':
                worker_ips = UserConfig().data.general.cluster.master_ip
                for ip in UserConfig().data.general.cluster.slaves:
                    worker_ips = worker_ips + ',' + ip
            cmd = ['bash', '{}/horovod/run_cluster_horovod_train.sh'.format(pwd_dir),
                   str(self.world_device_size), cf_file_remote, worker_ips]
        else:
            # Roma
            cmd = ['bash', '{}/horovod/run_horovod_train.sh'.format(pwd_dir),
                   str(self.world_device_size), cf_file_remote]
        proc = subprocess.Popen(cmd, env=os.environ)
        proc.wait()

    @property
    def world_device_size(self):
        """World device size is world size * device count in each world."""
        import torch
        world_size = UserConfig().data.env.world_size
        device_nums = torch.cuda.device_count()
        num_devices = world_size * device_nums
        return num_devices
