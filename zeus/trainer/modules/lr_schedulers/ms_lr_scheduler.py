# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Cosine annealing lr scheduler."""
from zeus.common import ClassFactory, ClassType
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore import Tensor
import numpy as np


@ClassFactory.register(ClassType.LR_SCHEDULER)
class MultiStepLR(LearningRateSchedule):
    """Multiple Step learning rate with warm up.

    :param milestones: list of decay epochs
    :type milestones: list of int
    :param decay_rates: list of decay rates
    :type decay_rates: list of float
    :param warmup: whether to warm up
    :type warmup: bool
    :param epoch_steps: steps in one epoch
    :type epoch_steps: int
    """

    def __init__(self, base_lr, milestones, gamma, total_epoch):
        """Initialize."""
        super(MultiStepLR, self).__init__()
        self.milestones = milestones
        self.base_lr = base_lr
        self.gamma = gamma
        self.total_epoch = total_epoch

    def construct(self, global_step):
        """Call lr scheduler class."""
        lr_each_step = []
        decay_step_index = [int(global_step * (self.milestones[i] / self.total_epoch)) for i in
                            range(len(self.milestones) + 1)]
        for i in range(global_step):
            if i < decay_step_index[0]:
                lr_each_step.append(self.base_lr)
            elif i < decay_step_index[1]:
                lr_each_step.append(self.base_lr * self.gamma)
            else:
                lr_each_step.append(self.base_lr * self.gamma * self.gamma)
        lr_each_step = np.array(lr_each_step).astype(np.float32)
        return Tensor(lr_each_step)
