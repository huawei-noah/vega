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

"""Defined faster rcnn detector."""
import tf_slim as slim
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.NETWORK)
class Regularizer(object):
    """Regularizer."""

    def __init__(self, desc):
        """Init ScopeGenerator.

        :param desc: config dict
        """
        self.model = None
        self.type = desc.type if 'type' in desc else None
        self.weight = desc.weight

    def get_real_model(self):
        """Get real model of regularizer."""
        if self.model:
            return self.model
        else:
            if self.type == 'l1_regularizer':
                self.model = slim.l1_regularizer(scale=float(self.weight))
            elif self.type == 'l2_regularizer':
                self.model = slim.l2_regularizer(scale=float(self.weight))
            else:
                self.model = None
                raise ValueError(
                    'Unknown regularizer type: {}'.format(self.type))

            return self.model
