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

"""The trainer program for SegmentationEA."""
import logging
import torch
import vega
from vega.common import ClassFactory, ClassType
from vega.metrics import calc_model_flops_params
from vega.trainer.callbacks import Callback

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class SegmentationEATrainerCallback(Callback):
    """Construct the trainer of Adelaide-EA."""

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.config = self.trainer.config
        if vega.is_npu_device():
            count_input = torch.FloatTensor(1, 3, 1024, 1024).npu()
        else:
            count_input = torch.FloatTensor(1, 3, 1024, 1024).cuda()
        flops_count, params_count = calc_model_flops_params(
            self.trainer.model, count_input)
        self.flops_count, self.params_count = flops_count * 1e-9, params_count * 1e-3
        logger.info("Flops: {:.2f} G, Params: {:.1f} K".format(self.flops_count, self.params_count))

    def after_epoch(self, epoch, logs=None):
        """Update flops and params."""
        summary_perfs = logs.get('summary_perfs', {})
        summary_perfs.update({'flops': self.flops_count, 'params': self.params_count})
        logs.update({'summary_perfs': summary_perfs})
