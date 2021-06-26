# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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

            # num_classes = ModelConfig.model_desc.backbone.n_class
            num_classes = ModelConfig.num_classes

            model = self.trainer.model
            out_features = num_classes

            # fix layers
            # for param in model.parameters():
            #     param.requires_grad = False

            # change head
            if "torch_vision_model" in ModelConfig.model_desc["modules"]:
                # torchvision
                import torch.nn as nn
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, out_features).cuda()
            else:
                # vega
                in_features = model.fc.in_features
                from vega.modules.operators import ops
                model.fc = ops.Linear(in_features=in_features, out_features=out_features).cuda()
                # TODO n_class
                ModelConfig.model_desc.backbone.n_class = num_classes
                logging.info("Model fine tuned successfully.")
