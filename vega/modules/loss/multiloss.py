# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""jdd_loss for task."""

from vega.modules.module import Module
from vega.common.class_factory import ClassType, ClassFactory


@ClassFactory.register(ClassType.LOSS)
class MultiLoss(Module):
    """Define Multi loss creator for base."""

    def __init__(self, *losses):
        """Initialize loss."""
        super(MultiLoss, self).__init__(*losses)
        self.loss_fn = []
        for loss in losses:
            self.loss_fn.append(loss)

    def call(self, output, target):
        """Sum all loss of predict and groundtruth."""
        outputs = None
        for model in self.loss_fn:
            if outputs is None:
                outputs = model(output, target)
            else:
                outputs = outputs + model(output, target)
        return outputs


@ClassFactory.register(ClassType.LOSS)
class SingleLoss(Module):
    """Select loss function from dict."""

    def __init__(self, key='loss'):
        super(SingleLoss, self).__init__()
        self.key = key

    def call(self, losses, targets=None):
        """Compute loss.

        :param inputs: predict data.
        :param targets: true data.
        :return:
        """
        if not isinstance(losses, dict):
            return losses
        return losses[self.key]
