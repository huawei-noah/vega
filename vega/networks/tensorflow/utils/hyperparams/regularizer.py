# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
