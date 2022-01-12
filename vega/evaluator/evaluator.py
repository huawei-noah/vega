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

"""Evaluate used to do evaluate process."""

import copy
import logging
import os
import glob
import vega
from vega.common import ClassFactory, ClassType
from vega.trainer.distributed_worker import DistributedWorker
from vega.trainer.utils import WorkerTypes
from vega.common import FileOps, Config
from vega.datasets import Adapter
from vega.model_zoo import ModelZoo
from vega.networks.model_config import ModelConfig
from vega.core.pipeline.conf import PipeStepConfig
from .conf import EvaluatorConfig

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
        self.saved_folder = self.get_local_worker_path(self.step_name, self.worker_id)
        if not self.model_desc:
            self.model_desc = self._get_model_desc()
        if not self.weights_file:
            if vega.is_torch_backend():
                self.weights_file = FileOps.join_path(self.saved_folder, 'model_{}.pth'.format(self.worker_id))
            elif vega.is_ms_backend():
                for file in os.listdir(self.saved_folder):
                    if file.endswith(".ckpt"):
                        self.weights_file = FileOps.join_path(self.saved_folder, file)
            elif vega.is_tf_backend():
                self.weights_file = FileOps.join_path(self.saved_folder, 'model_{}'.format(self.worker_id))
        if self.weights_file is not None and os.path.exists(self.weights_file):
            self.model = ModelZoo.get_model(self.model_desc, self.weights_file, is_fusion=self.config.is_fusion)
        else:
            logger.info("evalaute model without loading weights file")
            self.model = ModelZoo.get_model(self.model_desc, is_fusion=self.config.is_fusion)

    def _use_evaluator(self):
        """Check if use evaluator and get the evaluators.

        :return: if we used evaluator, and Evaluator classes
        :rtype: bool, (Evaluator, HostEvaluator, DloopEvaluator)
        """
        use_evaluator = False
        cls_evaluator_set = []
        if EvaluatorConfig.host_evaluator_enable:
            cls_host_evaluator = ClassFactory.get_cls(ClassType.HOST_EVALUATOR, EvaluatorConfig.host_evaluator.type)
            use_evaluator = True
            cls_evaluator_set.append(cls_host_evaluator)
        if EvaluatorConfig.device_evaluator_enable:
            cls_device_evaluator = ClassFactory.get_cls(ClassType.DEVICE_EVALUATOR,
                                                        EvaluatorConfig.device_evaluator.type)
            use_evaluator = True
            cls_evaluator_set.append(cls_device_evaluator)
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
        for cls in cls_evaluator_set:
            evaluator = cls(worker_info=self.worker_info)
            self.add_evaluator(evaluator)
        self._disable_host_latency()

    def _disable_host_latency(self):
        if len(self.sub_worker_list) < 2:
            return
        for sub_evaluator in self.sub_worker_list:
            if sub_evaluator.worker_type == WorkerTypes.HOST_EVALUATOR:
                sub_evaluator.config.evaluate_latency = False

    def _get_model_desc(self):
        model_desc = self.model_desc
        self.saved_folder = self.get_local_worker_path(self.step_name, self.worker_id)
        if not model_desc:
            if os.path.exists(FileOps.join_path(self.saved_folder, 'desc_{}.json'.format(self.worker_id))):
                model_config = Config(FileOps.join_path(self.saved_folder, 'desc_{}.json'.format(self.worker_id)))
                if "type" not in model_config and "modules" not in model_config:
                    model_config = ModelConfig.model_desc
                model_desc = model_config
            elif ModelConfig.model_desc_file is not None:
                desc_file = ModelConfig.model_desc_file
                desc_file = desc_file.replace("{local_base_path}", self.local_base_path)
                if ":" not in desc_file:
                    desc_file = os.path.abspath(desc_file)
                if ":" in desc_file:
                    local_desc_file = FileOps.join_path(
                        self.local_output_path, os.path.basename(desc_file))
                    FileOps.copy_file(desc_file, local_desc_file)
                    desc_file = local_desc_file
                model_desc = Config(desc_file)
                logger.info("net_desc:{}".format(model_desc))
            elif ModelConfig.model_desc is not None:
                model_desc = ModelConfig.model_desc
            elif ModelConfig.models_folder is not None:
                folder = ModelConfig.models_folder.replace("{local_base_path}", self.local_base_path)
                pattern = FileOps.join_path(folder, "desc_*.json")
                desc_file = glob.glob(pattern)[0]
                model_desc = Config(desc_file)
            elif PipeStepConfig.pipe_step.get("models_folder") is not None:
                folder = PipeStepConfig.pipe_step.get("models_folder").replace("{local_base_path}",
                                                                               self.local_base_path)
                desc_file = FileOps.join_path(folder, "desc_{}.json".format(self.worker_id))
                model_desc = Config(desc_file)
                logger.info("Load model from model folder {}.".format(folder))
        return model_desc
