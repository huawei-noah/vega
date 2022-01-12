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

"""Reduce Loss."""
import functools
import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """Reduce loss compute.

    :param loss: losses
    :param reduction: reduce funtion
    :return: loss
    """
    reduction_function = F._Reduction.get_enum(reduction)
    if reduction_function == 0:
        return loss
    elif reduction_function == 1:
        return loss.mean()
    elif reduction_function == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Weight reduce loss.

    :param loss: losses
    :param weight: weight
    :param reduction: reduce function
    :param avg_factor: avg factor
    :return: loss
    """
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Weight loss compute.

    :param loss_func: loss function
    :return: loss
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper
