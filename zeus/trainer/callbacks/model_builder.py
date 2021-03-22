# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ModelCheckpoint callback defination."""
import os
import glob
import logging
import zeus
from .callback import Callback
from zeus.common import FileOps, Config
from zeus.common import ClassFactory, ClassType
from zeus.networks.model_config import ModelConfig
from zeus.common.general import General
from zeus.model_zoo import ModelZoo

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class ModelBuilder(Callback):
    """Callback that saves the evaluated Performance."""

    def __init__(self):
        """Initialize ModelCheckpoint callback."""
        super(Callback, self).__init__()
        self.priority = 240

    def init_trainer(self, logs=None):
        """Set trainer object for current callback."""
        self.trainer.model = self._init_model()

    def _init_model(self):
        """Load model desc from save path and parse to model."""
        model = self.trainer.model
        if self.trainer.config.is_detection_trainer:
            model_desc = self.trainer.model_desc or self._get_model_desc()
        else:
            model_desc = self._get_model_desc()
        pretrained_model_file = self._get_pretrained_model_file()
        if not model:
            if not model_desc:
                raise Exception("Failed to Init model, can not get model description.")
            model = ModelZoo.get_model(model_desc, pretrained_model_file, ModelConfig.head)
        if model:
            self.trainer.model_desc = model.desc
            if zeus.is_torch_backend():
                import torch
                if self.trainer.use_cuda:
                    model = model.cuda()
                if General._parallel and General.devices_per_trainer > 1:
                    model = torch.nn.DataParallel(model)
        return model

    def _get_model_desc(self):
        model_desc = self.trainer.model_desc
        if not model_desc:
            if ModelConfig.model_desc_file is not None:
                desc_file = ModelConfig.model_desc_file
                desc_file = desc_file.replace("{local_base_path}", self.trainer.local_base_path)
                if ":" not in desc_file:
                    desc_file = os.path.abspath(desc_file)
                if ":" in desc_file:
                    local_desc_file = FileOps.join_path(
                        self.trainer.local_output_path, os.path.basename(desc_file))
                    FileOps.copy_file(desc_file, local_desc_file)
                    desc_file = local_desc_file
                model_desc = Config(desc_file)
                logger.info("net_desc:{}".format(model_desc))
            elif ModelConfig.model_desc is not None:
                model_desc = ModelConfig.model_desc
            elif ModelConfig.models_folder is not None:
                folder = ModelConfig.models_folder.replace("{local_base_path}", self.trainer.local_base_path)
                pattern = FileOps.join_path(folder, "desc_*.json")
                desc_file = glob.glob(pattern)[0]
                model_desc = Config(desc_file)
        return model_desc

    def _get_pretrained_model_file(self):
        model_file = self.trainer.config.kwargs.get("pretrained_model_file")
        if model_file:
            return model_file
        if ModelConfig.pretrained_model_file:
            model_file = ModelConfig.pretrained_model_file
            model_file = model_file.replace("{local_base_path}", self.trainer.local_base_path)
            model_file = model_file.replace("{worker_id}", str(self.trainer.worker_id))
            if ":" not in model_file:
                model_file = os.path.abspath(model_file)
            if ":" in model_file:
                local_model_file = FileOps.join_path(
                    self.trainer.local_output_path, os.path.basename(model_file))
                FileOps.copy_file(model_file, local_model_file)
                model_file = local_model_file
            return model_file
        else:
            return None
