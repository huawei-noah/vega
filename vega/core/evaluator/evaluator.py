# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Evaluate used to do evaluate process."""
import os
import copy
import logging
import torch
import pickle
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.trainer.distributed_worker import DistributedWorker
from vega.core.trainer.utils import WorkerTypes
from vega.core.common import FileOps
from vega.core.report import Report, ReportRecord
from vega.search_space.networks.network_desc import NetworkDesc

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.EVALUATOR)
class Evaluator(DistributedWorker):
    """Evaluator.

    :param worker_info: worker_info
    :type worker_info: dict, default to None
    """

    def __init__(self, worker_info=None):
        """Init Evaluator."""
        super(Evaluator, self).__init__()
        Evaluator.__worker_id__ = Evaluator.__worker_id__ + 1
        self._worker_id = Evaluator.__worker_id__
        self.worker_type = WorkerTypes.EVALUATOR
        self.worker_info = worker_info
        if worker_info is not None and "step_name" in worker_info and "worker_id" in worker_info:
            self.step_name = self.worker_info["step_name"]
            self.worker_id = self.worker_info["worker_id"]
        # main evalutors setting
        self.sub_worker_list = []
        self._init_evaluator()

    @property
    def size(self):
        """Return the size of current evaluator list."""
        return len(self.sub_worker_list)

    def add_evaluator(self, evaluator):
        """Add a sub-evaluator to this evaluator.

        :param evaluator: Description of parameter `evaluator`.
        :type evaluator: object,
        """
        if not isinstance(evaluator, DistributedWorker):
            return
        elif evaluator.worker_type is not None:
            sub_evaluator = copy.deepcopy(evaluator)
            sub_evaluator.worker_info = self.worker_info
            if self.worker_info is not None:
                sub_evaluator.step_name = self.worker_info["step_name"]
                sub_evaluator.worker_id = self.worker_info["worker_id"]
            self.sub_worker_list.append(sub_evaluator)
        return

    def set_worker_info(self, worker_info):
        """Set current evaluator's worker_info.

        :param worker_info: Description of parameter `worker_info`.
        :type worker_info: dict,
        """
        if worker_info is None:
            raise ValueError("worker_info should not be None type!")
        self.worker_info = worker_info
        self.step_name = self.worker_info["step_name"]
        self.worker_id = self.worker_info["worker_id"]

        for sub_evaluator in self.sub_worker_list:
            sub_evaluator.worker_info = self.worker_info
            sub_evaluator.step_name = self.worker_info["step_name"]
            sub_evaluator.worker_id = self.worker_info["worker_id"]
        return

    def _init_dataloader(self, mode, loader=None):
        """Init dataloader."""
        if loader is not None:
            return loader
        dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
        dataset = dataset_cls(mode=mode)
        return dataset.dataloader

    def _load_pretrained_model(self):
        if self.model is None:
            return
        if self.config.pretrained_model_file is not None:
            model_file = self.config.pretrained_model_file
            model_file = os.path.abspath(model_file)
            ckpt = torch.load(model_file)
            self.model.load_state_dict(ckpt)
            return

    def load_checkpoint(self, worker_id=None, step_name=None, saved_folder=None):
        """Load checkpoint."""
        if saved_folder is None:
            if worker_id is None:
                worker_id = self.worker_id
            if step_name is None:
                step_name = self.step_name
            saved_folder = self.get_local_worker_path(step_name, worker_id)
        checkpoint_file = FileOps.join_path(
            saved_folder, self.checkpoint_file_name)
        model_pickle_file = FileOps.join_path(
            saved_folder, self.model_pickle_file_name)
        if not (os.path.isfile(checkpoint_file)):
            checkpoint_file = FileOps.join_path(
                saved_folder, str(self.worker_id), self.checkpoint_file_name)
        if not (os.path.isfile(model_pickle_file)):
            model_pickle_file = FileOps.join_path(
                saved_folder, str(self.worker_id), self.model_pickle_file_name)
        try:
            with open(model_pickle_file, 'rb') as f:
                model = pickle.load(f)
                ckpt = torch.load(
                    checkpoint_file, map_location=torch.device('cpu'))
                model.load_state_dict(ckpt['weight'])
                if self.config.cuda:
                    model = model.cuda()
                self.model = model
        except Exception:
            logging.info(
                'Checkpoint file is not existed, use default model now.')
            return

    def _use_evaluator(self):
        """Check if use evaluator and get the evaluators.

        :return: if we used evaluator, and Evaluator classes
        :rtype: bool, (Evaluator, GpuEvaluator, DloopEvaluator)
        """
        use_evaluator = False
        cls_evaluator_set = []
        try:
            cls_gpu_evaluator = ClassFactory.get_cls(ClassType.GPU_EVALUATOR)
            use_evaluator = True
            cls_evaluator_set.append(cls_gpu_evaluator)
        except Exception as e:
            logger.warning("evaluator not been set. {}".format(str(e)))
        try:
            cls_hava_d_evaluator = ClassFactory.get_cls(
                ClassType.HAVA_D_EVALUATOR)
            use_evaluator = True
            cls_evaluator_set.append(cls_hava_d_evaluator)
        except:
            pass
        try:
            cls_davinci_mobile_evaluator = ClassFactory.get_cls(
                ClassType.DAVINCI_MOBILE_EVALUATOR)
            use_evaluator = True
            cls_evaluator_set.append(cls_davinci_mobile_evaluator)
        except:
            pass
        return use_evaluator, cls_evaluator_set

    def _init_evaluator(self):
        """Do evaluate stuff.

        :param finished_trainer_info: the finished trainer info
        :type: list or dict

        """
        use_evaluator, cls_evaluator_set = self._use_evaluator()
        if not use_evaluator:
            return
        record = Report().receive(self.step_name, self.worker_id)
        model_desc = record.desc
        model = NetworkDesc(model_desc).to_model()
        for cls in cls_evaluator_set:
            evaluator = cls(worker_info=self.worker_info, model=model)
            self.add_evaluator(evaluator)
