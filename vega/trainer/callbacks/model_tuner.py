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

"""ModleTuner callback defination."""
import logging
import vega
from vega.trainer.callbacks.callback import Callback
from vega.common import ClassFactory, ClassType
from vega.networks.model_config import ModelConfig


@ClassFactory.register(ClassType.CALLBACK)
class ModelTuner(Callback):
    """Callback that saves the evaluated Performance."""

    def __init__(self):
        """Initialize ModleTuner callback."""
        super(ModelTuner, self).__init__()
        self.priority = 250

    def init_trainer(self, logs=None):
        """Init model. Change head and Fix layers."""
        self._reset_classifier_model()

    def _reset_classifier_model(self):
        if vega.is_torch_backend():
            num_classes = ModelConfig.num_classes

            model = self.trainer.model
            out_features = num_classes
            if "torch_vision_model" in ModelConfig.model_desc["modules"]:
                import torch.nn as nn
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, out_features).cuda()
            else:
                in_features = model.fc.in_features
                from vega.modules.operators import ops
                model.fc = ops.Linear(in_features=in_features, out_features=out_features).cuda()
                ModelConfig.model_desc.backbone.n_class = num_classes
                logging.info("Model fine tuned successfully.")
