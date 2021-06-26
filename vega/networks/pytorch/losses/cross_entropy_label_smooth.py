# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Cross Entropy Label Smooth Loss."""
import torch
import torch.nn as nn
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LOSS)
class CrossEntropyLabelSmooth(nn.Module):
    """Class of Cross Entropy Label Smooth Loss.

    :param num_classes: number of classes
    :type num_classes: int
    :param epsilon: label smooth coefficient
    :type epsilon: float
    """

    def __init__(self, num_classes, epsilon):
        """Init CrossEntropyLabelSmooth."""
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """Forward function."""
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
