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
class SREATrainerCallback(Callback):
    """Construct the trainer of ESR-EA."""

    def make_batch(self, batch):
        """Make batch for each training step."""
        input = batch["LR"]
        target = batch["HR"]
        if self.trainer.config.cuda and not self.trainer.config.is_detection_trainer:
            input = input.cuda() / 255.0
            target = target.cuda() / 255.0
        return input, target
