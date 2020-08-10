# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
