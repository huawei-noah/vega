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

"""Smooth L1 Loss."""
import torch
from vega.modules.module import Module
from vega.common import ClassType, ClassFactory
from .reduce_loss import weighted_loss


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth l1 loss.

    :param pred: predict
    :param target: target
    :param beta: beta
    :return: loss
    """
    if beta > 0 and pred.size() == target.size() and target.numel() > 0:
        diff = torch.abs(pred - target)
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        return loss
    else:
        raise ValueError('Failed to calculate smooth l1 loss.')


@ClassFactory.register(ClassType.NETWORK)
class SmoothL1Loss(Module):
    """Smooth L1 Loss."""

    def __init__(self, desc):
        """Init smooth l1 loss.

        :param desc: config dict
        """
        super(SmoothL1Loss, self).__init__()
        self.beta = desc['beta'] if 'beta' in desc else 1.0
        self.reduction = desc['reduction'] if 'reduction' in desc else 'mean'
        self.loss_weight = desc['loss_weight'] if 'loss_weight' in desc else 1.0

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward compute.

        :param pred: predict
        :param target: target
        :param weight: weight
        :param avg_factor: avg factor
        :param reduction_override: reduce override
        :return: loss
        """
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if target.numel() > 0:
            loss_bbox = self.loss_weight * smooth_l1_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
            return loss_bbox
        else:
            return torch.FloatTensor([0.0]).cuda()
