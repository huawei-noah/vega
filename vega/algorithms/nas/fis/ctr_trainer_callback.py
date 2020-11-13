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
from zeus.common import ClassFactory, ClassType
from zeus.trainer.callbacks import Callback

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
        if self.config.cuda:
            input, target = input.cuda(), target.cuda()
        return (input, target)
