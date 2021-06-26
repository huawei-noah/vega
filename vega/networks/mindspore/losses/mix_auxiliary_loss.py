# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Mix Auxiliary Loss."""
import mindspore.nn as nn
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LOSS)
class MixAuxiliaryLoss(nn.Cell):
    """Class of Mix Auxiliary Loss.

    :param aux_weight: auxiliary loss weight
    :type aux_weight: float
    :loss_base: base loss function
    :loss_base: str
    """

    def __init__(self, aux_weight, loss_base):
        """Init MixAuxiliaryLoss."""
        super(MixAuxiliaryLoss, self).__init__()
        self.aux_weight = aux_weight
        loss_base_cp = loss_base.copy()
        loss_base_name = loss_base_cp.pop('type')
        self.loss_fn = ClassFactory.get_cls('trainer.loss', loss_base_name)(**loss_base_cp['params'])

    def construct(self, outputs, targets):
        """Loss forward function."""
        loss0 = self.loss_fn(outputs[0], targets)
        loss1 = self.loss_fn(outputs[1], targets)
        return loss0 + self.aux_weight * loss1
