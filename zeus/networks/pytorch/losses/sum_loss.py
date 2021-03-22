# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Sum_loss for detection task."""
import torch
from torch import nn
from collections import OrderedDict
from zeus.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.LOSS)
class SumLoss(nn.Module):
    """Calculate sum of input losses."""

    def __init__(self):
        """Init SumLoss."""
        super(SumLoss, self).__init__()

    def forward(self, input, target=None):
        """Calculate sum of input losses.

        :param input: dict of losses.
        :type input: dict
        :param target: `target` Tensor, default None.
        :type target: type torch.Tensor
        :return: return sum of losses.
        :rtype: torch.Tensor

        """
        losses = input
        if not isinstance(losses, dict) and not isinstance(losses, OrderedDict):
            return None
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    '{} is not a tensor or list of tensors'.format(loss_name))

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        # Debug
        """
        if loss > 100:
            logging.error(str(losses))
            import os
            os._exit()
        """
        return loss
