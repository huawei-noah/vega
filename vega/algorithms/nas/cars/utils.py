# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Util functions."""
import numpy as np


class AverageMeter(object):
    """This is a meter class to calculate average values."""

    def __init__(self):
        """Construct method."""
        self.reset()

    def reset(self):
        """Reset the meter."""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """Update the meter."""
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def eval_model_parameters(model):
    """Calculate number of parameters in million (M) for a model.

    :param model: A model
    :type model: nn.Module
    :return: The number of parameters
    :rtype: Float
    """
    return np.sum(v.numel() for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6
