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

"""Data parallel callback."""

import logging
import torch
import vega
from vega.common import ClassFactory, ClassType
from vega.common.general import General
from .callback import Callback

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
