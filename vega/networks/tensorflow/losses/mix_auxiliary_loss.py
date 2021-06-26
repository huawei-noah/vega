# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Mix Auxiliary Loss."""
import importlib
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LOSS)
class MixAuxiliaryLoss(object):
    """Class of Mix Auxiliary Loss.

    :param aux_weight: auxiliary loss weight
    :type aux_weight: float
    :loss_base: base loss function
    :loss_base: str
    """

    def __init__(self, aux_weight, loss_base):
        """Init MixAuxiliaryLoss."""
        self.aux_weight = aux_weight
        loss_base_cp = loss_base.copy()
        loss_base_name = loss_base_cp.pop('type')
        if ClassFactory.is_exists('trainer.loss', loss_base_name):
            loss_class = ClassFactory.get_cls('trainer.loss', loss_base_name)
        else:
            loss_class = getattr(importlib.import_module('tensorflow.losses'), loss_base_name)
        self.loss_fn = loss_class(**loss_base_cp['params'])

    def __call__(self, logits, labels):
        """Loss forward function."""
        if logits.get_shape()[0] != 2:
            raise Exception('outputs length must be 2')
        loss0 = self.loss_fn(logits=logits[0], labels=labels)
        loss1 = self.loss_fn(logits=logits[1], labels=labels)
        return loss0 + self.aux_weight * loss1
