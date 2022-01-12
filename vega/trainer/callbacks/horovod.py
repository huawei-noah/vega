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
import vega
from vega.common import ClassFactory, ClassType
from .callback import Callback

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class Horovod(Callback):
    """Callback that saves the evaluated Performance."""

    def __init__(self):
        """Initialize ModelCheckpoint callback."""
        super(Horovod, self).__init__()
        self.priority = 260

    def before_train(self, logs=None):
        """Be called before the training process."""
        if not self.trainer.horovod:
            return
        if vega.is_torch_backend():
            self._init_torch()

    def _init_torch(self):
        import torch
        import horovod.torch as hvd
        hvd.broadcast_parameters(self.trainer.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.trainer.optimizer, root_rank=0)
        self.trainer._average_metrics = self._average_metrics

    def _average_metrics(self, metrics_results):
        import torch
        import horovod.torch as hvd
        for key, value in metrics_results.items():
            tensor = torch.tensor(value)
            avg_tensor = hvd.allreduce(tensor, name=key)
            metrics_results[key] = avg_tensor.item()
        return metrics_results
