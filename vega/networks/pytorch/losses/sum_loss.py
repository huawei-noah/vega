# -*- coding: utf-8 -*-

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

"""Sum_loss for detection task."""

import os
import logging
from collections import OrderedDict
import torch
from torch import nn
from vega.common import ClassType, ClassFactory
from vega.common import FileOps


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

        init_loss = [_value for _key, _value in log_vars.items() if 'loss' in _key]

        if hasattr(self, "dynamic_loss_weight"):
            loss_save = [float(_value.detach().cpu().numpy()) for _value in init_loss]
            save_file = os.path.join(self.save_path, "muti_loss.pkl")
            FileOps.dump_pickle(loss_save, save_file)

            if len(self.dynamic_loss_weight) != len(init_loss):
                logging.error("The length of the loss must be same with the length of the weight, but got {} and {}"
                              .format(len(init_loss), len(self.dynamic_loss_weight)))
            weighted_loss = [self.dynamic_loss_weight[i] * init_loss[i] for i in range(len(init_loss))]

            sum_loss = sum(weighted_loss)
        else:
            sum_loss = sum(init_loss)
        return sum_loss

    def adaptive_muti_loss(self, save_path, weight):
        """Set adaptive muti loss params."""
        self.save_path = save_path
        self.dynamic_loss_weight = weight
