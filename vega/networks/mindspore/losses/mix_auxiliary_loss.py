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
