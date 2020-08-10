# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The trainer program for ESR_EA."""
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.trainer.callbacks import Callback


@ClassFactory.register(ClassType.CALLBACK)
class ESRTrainerCallback(Callback):
    """Construct the trainer of ESR-EA."""

    def before_train(self, epoch, logs=None):
        """Be called before the training process."""
        self.config = self.trainer.config
        # Use own save checkpoint and save performance function
        self.trainer.auto_save_ckpt = False
        self.trainer.auto_save_perf = False
        # This part is tricky and

    def make_batch(self, batch):
        """Make batch for each training step."""
        input = batch["LR"]
        target = batch["HR"]
        if self.config.cuda:
            input = input.cuda()
            target = target.cuda()
        return input, target
