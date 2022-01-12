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

"""ModelCheckpoint callback defination."""

import logging
import vega
from vega.common import Config
from vega.common import ClassFactory, ClassType
from vega.networks.model_config import ModelConfig
from vega.model_zoo import ModelZoo
from .callback import Callback

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class ModelBuilder(Callback):
    """Callback that saves the evaluated Performance."""

    def __init__(self):
        """Initialize ModelCheckpoint callback."""
        super(ModelBuilder, self).__init__()
        self.priority = 200

    def init_trainer(self, logs=None):
        """Set trainer object for current callback."""
        model = self.trainer.model
        if not model:
            model = self._init_model()
        if hasattr(model, "desc"):
            self.trainer.model_desc = model.desc
        self.trainer.model = self._set_device(model)

    def _init_model(self):
        """Load model desc from save path and parse to model."""
        config = Config(ModelConfig().to_dict())
        if self.trainer.model_desc:
            config.model_desc = self.trainer.model_desc
        if not config.model_desc:
            raise Exception("Failed to Init model, can not get model description.")
        if self.trainer.load_weights_file:
            config.pretrained_model_file = self.trainer.config.kwargs.get(
                "pretrained_model_file") or config.pretrained_model_file
        return ModelZoo.get_model(**config)

    def _set_device(self, model):
        if vega.is_torch_backend():
            if vega.is_gpu_device():
                model = model.cuda()
            elif vega.is_npu_device():
                model = model.to(vega.get_devices())
        return model
