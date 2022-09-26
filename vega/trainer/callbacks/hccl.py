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

import os
import logging
import vega
from vega.common import ClassFactory, ClassType
from vega.common.general import General
from .callback import Callback

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
        import torch.distributed as dist
        logger.info("init HCCL")
        dist.init_process_group(
            backend='hccl',
            init_method=f"tcp://{General.cluster.hccl_server_ip}:{General.cluster.hccl_port}",
            world_size=self.trainer.num_workers,
            rank=self.trainer.rank_id)

    def _init_ms_trainer(self):
        from mindspore import context
        from mindspore.context import ParallelMode
        from mindspore.communication.management import init

        logger.info("init HCCL")
        context.set_auto_parallel_context(device_num=int(os.getenv('RANK_SIZE', '1')), parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
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
