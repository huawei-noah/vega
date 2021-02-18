# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""jdd_loss for task."""
from inspect import isclass
from zeus.modules.module import Module
from zeus.common.class_factory import ClassType, ClassFactory


@ClassFactory.register(ClassType.LOSS)
class MultiLoss(Module):
    """Define Multi loss creator for base."""

    def __init__(self, *losses):
        """Initialize loss."""
        super(MultiLoss, self).__init__(*losses)
        for loss in losses:
            name = loss.__class__.__name__ if isclass(loss) else None
            self._modules[name] = loss

    def __call__(self, output, target):
        """Sum all loss of predict and groundtruth."""
        outputs = None
        for model in list(self._modules.values()):
            if outputs is None:
                outputs = model(output, target)
            else:
                outputs = outputs + model(output, target)
        return outputs
