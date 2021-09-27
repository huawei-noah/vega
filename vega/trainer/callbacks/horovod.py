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
import vega
from .callback import Callback
from vega.common import ClassFactory, ClassType

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
        # elif vega.is_tf_backend():
        #     self._init_tf()

    def _init_torch(self):
        import torch
        import horovod.torch as hvd
        hvd.broadcast_parameters(self.trainer.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.trainer.optimizer, root_rank=0)
        # torch.cuda.set_device(hvd.local_rank())
        self.trainer._average_metrics = self._average_metrics

    # def _init_tf(self):
    #     import horovod.tensorflow as hvd
    #     # hvd.init()
    #     # TODO horovod tf
    #     self.trainer.sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

    def _average_metrics(self, metrics_results):
        import torch
        import horovod.torch as hvd
        for key, value in metrics_results.items():
            tensor = torch.tensor(value)
            avg_tensor = hvd.allreduce(tensor, name=key)
            metrics_results[key] = avg_tensor.item()
        return metrics_results
