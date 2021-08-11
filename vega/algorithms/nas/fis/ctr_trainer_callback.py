# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Base CTR model TrainerCallback."""

import logging
import vega
from vega.common import ClassFactory, ClassType
from vega.trainer.callbacks import Callback

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class CtrTrainerCallback(Callback):
    """CtrTrainerCallback module."""

    def __init__(self):
        """Constuct CtrTrainerCallback class."""
        super(CtrTrainerCallback, self).__init__()

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.config = self.trainer.config

    def make_batch(self, batch):
        """
        Make a batch data for ctr trainer.

        :param batch: a batch data
        :return: batch data, seperate input and target
        """
        input, target = batch
        if vega.is_gpu_device():
            input, target = input.cuda(), target.cuda()
        elif vega.is_npu_device():
            input, target = input.to(vega.get_devices()), target.to(vega.get_devices())
        return (input, target)
