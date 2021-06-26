# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""FocalLoss for unbalanced data."""
from vega.modules.operators import ops
from vega.modules.module import Module
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.LOSS)
class F1Loss(Module):
    """F1 Loss for unbalanced data."""

    def __init__(self, epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon

    def call(self, inputs, targets):
        """Compute loss.

        :param inputs: predict data.
        :param targets: true data.
        :return:
        """
        y_true = ops.to(ops.one_hot(targets, 2, ), 'float32')
        y_pred = ops.softmax(inputs, dim=1)

        tp = ops.reduce_sum(y_true * y_pred, dtype='float32')
        # tn = ops.reduce_sum(((1 - y_true) * (1 - y_pred)), dtype='float32')
        fp = ops.reduce_sum(((1 - y_true) * y_pred), dtype='float32')
        fn = ops.reduce_sum((y_true * (1 - y_pred)), dtype='float32')

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = ops.clamp(f1, min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()
