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
from vega.common.general import General

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class Hccl(Callback):
    """Callback that saves the evaluated Performance."""

    def __init__(self):
        """Initialize ModelCheckpoint callback."""
        super(Hccl, self).__init__()
        self.priority = 260

    def init_trainer(self, logs=None):
        """Set trainer object for current callback."""
        if not self.trainer.hccl:
            return

        if vega.is_torch_backend():
            self._init_pytorch_trainer()
        if vega.is_ms_backend():
            self._init_ms_trainer()

    def _init_pytorch_trainer(self):
        import torch
        import torch.distributed as dist
        logger.info("init HCCL")
        model = self.trainer.model
        dist.init_process_group(
            backend='hccl',
            init_method=f"tcp://{General.cluster.hccl_server_ip}:{General.cluster.hccl_port}",
            world_size=self.trainer.num_workers,
            rank=self.trainer.rank_id)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.trainer.device_id],
            broadcast_buffers=General.cluster.enable_broadcast_buffers)
        self.trainer.model = model

    def _init_ms_trainer(self):
        from mindspore import context
        from mindspore.context import ParallelMode
        from mindspore.communication.management import init

        logger.info("init HCCL")
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        init()

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        if not vega.is_torch_backend() or not self.trainer.hccl:
            return
        if self.trainer.sampler is not None:
            self.trainer.sampler.set_epoch(epoch)

    def after_train(self, logs=None):
        """Stop session."""
        if self.trainer.hccl and vega.is_tf_backend():
            self.trainer.sess.run(self.trainer.npu_shutdown)
            self.trainer.sess.close()
