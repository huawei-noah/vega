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
import zeus
from zeus.common import ClassFactory, ClassType
from zeus.trainer.distributed_worker import DistributedWorker
from zeus.trainer.utils import WorkerTypes
from zeus.common import FileOps
from zeus.report import Report
from zeus.datasets import Adapter
from .conf import EvaluatorConfig
from zeus.model_zoo import ModelZoo
from zeus.networks.model_config import ModelConfig

if zeus.is_torch_backend():
    import torch
elif zeus.is_tf_backend():
    pass

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
        self.model_desc = None
        self.weights_file = None
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
        dataloader = Adapter(dataset).loader
        return dataloader

    def load_model(self):
        """Load model."""
        if not self.model_desc and not self.weights_file:
            self.saved_folder = self.get_local_worker_path(self.step_name, self.worker_id)
            self.model_desc = FileOps.join_path(self.saved_folder, 'desc_{}.json'.format(self.worker_id))
            if zeus.is_torch_backend():
                self.weights_file = FileOps.join_path(self.saved_folder, 'model_{}.pth'.format(self.worker_id))
            elif zeus.is_torch_backend():
                self.weights_file = FileOps.join_path(self.saved_folder, 'model_{}.ckpt'.format(self.worker_id))
        if 'modules' not in self.model_desc:
            self.model_desc = ModelConfig.model_desc
        self.model = ModelZoo.get_model(self.model_desc, self.weights_file)

    def _use_evaluator(self):
        """Check if use evaluator and get the evaluators.

        :return: if we used evaluator, and Evaluator classes
        :rtype: bool, (Evaluator, GpuEvaluator, DloopEvaluator)
        """
        use_evaluator = False
        cls_evaluator_set = []
        if EvaluatorConfig.gpu_evaluator_enable:
            cls_gpu_evaluator = ClassFactory.get_cls(ClassType.GPU_EVALUATOR, "GpuEvaluator")
            use_evaluator = True
            cls_evaluator_set.append(cls_gpu_evaluator)
        if EvaluatorConfig.davinci_mobile_evaluator_enable:
            cls_davinci_mobile_evaluator = ClassFactory.get_cls(
                ClassType.DAVINCI_MOBILE_EVALUATOR, "DavinciMobileEvaluator")
            use_evaluator = True
            cls_evaluator_set.append(cls_davinci_mobile_evaluator)
        # TODO HAVA_D_EVALUATOR
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
        for cls in cls_evaluator_set:
            evaluator = cls(worker_info=self.worker_info, model_desc=model_desc)
            self.add_evaluator(evaluator)
