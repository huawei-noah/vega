# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Data parallel callback."""

import logging
import torch
import vega
from .callback import Callback
from vega.common import ClassFactory, ClassType
from vega.common.general import General

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class DdpTorch(Callback):
    """Callback that saves the evaluated Performance."""

    def __init__(self):
        """Initialize ModelCheckpoint callback."""
        super(DdpTorch, self).__init__()
        self.priority = 260

    def before_train(self, logs=None):
        """Be called before the training process."""
        if not vega.is_torch_backend() or not vega.is_gpu_device():
            return
        if not General._parallel or General.devices_per_trainer <= 1:
            return
        self.trainer.model = torch.nn.DataParallel(self.trainer.model)
