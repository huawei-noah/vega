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
